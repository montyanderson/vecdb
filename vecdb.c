#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "json.h"

typedef float number_t;
typedef number_t* vector_t;

typedef struct { // main state type
    char *path;
    size_t dimensions;
    size_t label_size;

    // fixed memory allocation space
    vector_t input_vector;
    vector_t hot_vector;
    char *io_buffer;
} state_t;

#define IO_BUFFER_SIZE (1024 * 1024)

#define get_vector_size(state) ((state)->dimensions * sizeof(number_t))
#define get_chunk_size(state) (get_vector_size((state)) + (state)->label_size)

/* io */

// read stdin into io_buffer
void *io_read_stdin(state_t *state) {
    assert(fgets(state->io_buffer, IO_BUFFER_SIZE, stdin) != NULL);
}

// parse json from io_buffer and put it in input_vector
void io_parse_json_vector(
    state_t *state
) {
    struct json_value_s *json = json_parse(
        state->io_buffer,
        strlen(state->io_buffer)
    );

    struct json_array_s* array = json_value_as_array(json);
    assert(array != NULL);
    assert(array->length == state->dimensions);

    size_t i = 0;

    for(
        struct json_array_element_s *element = array->start;
        element->next != NULL;
        element = element->next
    ) {
        struct json_value_s *value = element->value;
        assert(value->type == json_type_number);

        struct json_number_s *number = json_value_as_number(value);
        assert(number != NULL);

        // convert character slice into null terminated c string
        char number_str[1024];
        assert(number->number_size < sizeof number_str - 1);

        strncpy(number_str, number->number, number->number_size);
        number_str[number->number_size] = '\0';

        state->input_vector[i++] = atof(number_str);
    }
}

// write out io_buffer to db file
void io_db_append_chunk(
    state_t *state
) {
    FILE *file = fopen(state->path, "a");
    assert(file != NULL);

    assert(
        fwrite(
            state->io_buffer,
            1,
            get_chunk_size(state),
            file
        ) == get_chunk_size(state)
    );

    assert(fclose(file) == 0);
}

// turn vector + label into a serialized chunk
void io_serialize_chunk(state_t *state, char *label) {
    // wipe io_buffer
    memset(state->io_buffer, 0, get_chunk_size(state));

    // copy vector data
    memcpy(state->io_buffer, state->input_vector, get_vector_size(state));

    // copy vector label
    size_t label_len = (size_t) strlen(label);
    assert(label_len < state->label_size - 1);

    memcpy(state->io_buffer + get_vector_size(state), label, label_len);
}

typedef struct {
    char *mapping;
    size_t length; // length of database in chunks
} database_t;

// get read-only mapping of database file
database_t io_map_database(state_t *state) {
    int fd = open(state->path, O_RDONLY);
    assert(fd != -1);

    struct stat sb;
    assert(fstat(fd, &sb) != -1);

    database_t database;

    database.length = sb.st_size / get_chunk_size(state);
    assert(sb.st_size % get_chunk_size(state) == 0);

    database.mapping = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    assert(database.mapping != MAP_FAILED);

    return database;
}

// get a pointer to the beginning of a chunk – the vector
inline vector_t database_get_chunk(
    state_t *state,
    database_t *database,
    size_t i
) {
    return (vector_t) (
        database->mapping +
            (i * get_chunk_size(state))
    );
}

// get a pointer to the second part of a chunk – the label
inline vector_t database_get_chunk_label(
    state_t *state,
    database_t *database,
    size_t i
) {
    return (char *) (
        database->mapping + 
            (i * get_chunk_size(state)) + 
                get_vector_size(state)
    );
}

/* calculations */

float get_distance_basic(size_t dimensions, float *a, float *b) {
    float sum = 0.0;

    for (size_t i = 0; i < dimensions; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return __builtin_sqrtf(sum);
}

#define VECDB_SIMD 0
#define get_distance get_distance_basic

#ifdef __ARM_NEON
#include <arm_neon.h>

float get_distance_arm_simd(size_t dimensions, float *a, float *b) {
    assert(dimensions % 4 == 0);

    float sum = 0.0;

    for (int i = 0; i < dimensions; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);

        // vc = va - vb
        float32x4_t vc = vsubq_f32(va, vb);

        // vc = vc * vc
        vc = vmulq_f32(vc, vc);

        // vd = sum(vc)
        float32x2_t vd = vadd_f32(vget_low_f32(vc), vget_high_f32(vc));
        vd = vpadd_f32(vd, vd);

        sum += vget_lane_f32(vd, 0);
    }

    return  __builtin_sqrtf(sum);
}

#define get_distance get_distance_arm_simd
#define VECDB_SIMD 1
#endif


/* commands */

void command_add(state_t *state, int argc, char *argv[]) {
    io_read_stdin(state);
        // read stdin into state->io_buffer

    io_parse_json_vector(state);
        // parse state->io_buffer as json into state->input_vector

    char *label = argv[2];
    printf("Writing vector '%s'...\n", label);

    io_serialize_chunk( // turn state->input_vector into serialized state->io_buffer
        state,
        label
    );

    io_db_append_chunk( // state->io_buffer to file
        state
    );
}

void command_list(state_t *state, int argc, char *argv[]) {
    database_t db = io_map_database(state);

    for(size_t i = 0; i < db.length; i++) {
        printf("%s\n", database_get_chunk_label(state, &db, i));
    }
}

struct search_result_s {
    number_t distance;
    char *label;
};

void command_search(state_t *state, int argc, char*argv[]) {
    io_read_stdin(state);
        // read stdin into state->io_buffer

    io_parse_json_vector(state);
        // parse state->io_buffer as json into state->input_vector
    
    database_t db = io_map_database(state);

    const size_t results_length = 10;
    struct search_result_s results[results_length];

    for(size_t i = 0; i < db.length; i++) {
        // copy vector from database into the aligned space for comparison
        memcpy(
            state->hot_vector,
            database_get_chunk(state, &db, i),
            get_vector_size(state)
        );

        number_t distance = get_distance(
            state->dimensions,
            state->input_vector,
            state->hot_vector
        );
        
        char *label = database_get_chunk_label(state, &db, i);

        printf("%s\n\t%f\n", label, distance);
    }
}

int main(int argc, char *argv[]) {
    assert(sizeof(number_t) == 4); // better be fp32
    // set state params
    state_t state;

    state.path = getenv("VECDB_PATH");
    assert(state.path != NULL);

    char *dimensions_string = getenv("VECDB_DIMENSIONS");
    assert(state.path != NULL);

    state.dimensions = strtoul(dimensions_string, 0, 10);
    assert(state.dimensions > 0 && state.dimensions < UINT32_MAX);

    char *label_size_string = getenv("VECDB_LABEL_SIZE");
    assert(label_size_string != NULL);

    state.label_size = strtoul(label_size_string, 0, 10);
    assert(state.label_size > 0 && state.label_size < UINT32_MAX);

    fprintf(stderr, "[vecdb] float type = fp%d\n", sizeof(number_t) * 8);
    fprintf(stderr, "[vecdb] simd? = %s\n", VECDB_SIMD ? "true" : "false");
    fprintf(stderr, "[vecdb] state.path = '%s'\n", state.path);
    fprintf(stderr, "[vecdb] state.dimensions = '%d'\n", state.dimensions);
    fprintf(stderr, "[vecdb] state.label_size = '%d'\n", state.label_size);
    fprintf(stderr, "\n");

    size_t total_allocation =
        get_vector_size(&state) + // input vec
        get_vector_size(&state) + // hot vec
        IO_BUFFER_SIZE;

    // allocate single fixed-size block of memory
    char *memory = malloc(total_allocation);
    assert(memory != NULL);

    memset(memory, 0, total_allocation);

    // slice up
    state.input_vector = memory;
    state.hot_vector = memory + get_vector_size(&state);
    state.io_buffer = memory + get_vector_size(&state) + get_vector_size(&state);

    if(strcmp(argv[1], "add") == 0) {
        command_add(&state, argc, argv);
    } else if(strcmp(argv[1], "list") == 0) {
        command_list(&state, argc, argv);
    } else if(strcmp(argv[1], "search") == 0) {
        command_search(&state, argc, argv);
    } else {
        printf("Unknown subcommand '%s'!\n", argv[1]);
    }

    return 0;
}
