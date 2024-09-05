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
#include <math.h> // doesn't need linking
#include <time.h>

#include "json.h"

const char *vecdb = "[\033[1m\x1b[31mv\x1b[32me\x1b[33mc\x1b[34md\x1b[35mb\x1b[0m]";

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
static inline vector_t database_get_chunk(
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
static inline vector_t database_get_chunk_label(
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

/* non-simd get_distance() */

float get_distance_basic(size_t dimensions, float *a, float *b) {
    float sum = 0.0;

    for (size_t i = 0; i < dimensions; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }

    return __builtin_sqrtf(sum);
}

#define VECDB_SIMD "generic"
#define get_distance get_distance_basic

/* arm neon 128-bit get_distance() */

#ifdef __ARM_NEON
#include <arm_neon.h>

float get_distance_arm_128(size_t dimensions, float *a, float *b) {
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

#define get_distance get_distance_arm_128
#define VECDB_SIMD "neon-128"
#endif

/* x86 avr 256-bit get_distance() */

#ifdef __AVX__
#include <immintrin.h>

static inline float get_distance_avx_256(size_t dimensions, float *a, float *b) {
    assert(dimensions % 8 == 0);  // AVX processes 8 floats at a time

    __m256 sum_vec = _mm256_setzero_ps();  // Initialize sum vector to zero

    for (size_t i = 0; i < dimensions; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);  // Load 8 floats from array a
        __m256 vb = _mm256_loadu_ps(&b[i]);  // Load 8 floats from array b

        // vc = va - vb
        __m256 vc = _mm256_sub_ps(va, vb);

        // vc = vc * vc
        vc = _mm256_mul_ps(vc, vc);

        // Accumulate the sum of squares
        sum_vec = _mm256_add_ps(sum_vec, vc);
    }

    // Sum the elements of sum_vec
    __m128 vlow  = _mm256_castps256_ps128(sum_vec);
    __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
    vlow  = _mm_add_ps(vlow, vhigh);

    __m128 sum128 = _mm_hadd_ps(vlow, vlow);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float sum = _mm_cvtss_f32(sum128);

    return sqrtf(sum);
}

#undef get_distance
#define get_distance get_distance_avx_256
#undef VECDB_SIMD
#define VECDB_SIMD "avx-256"
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

/*
inline void search_insert_contender(
	struct search_result_s *results,
	size_t results_length,
	struct search_result_s *contender
) {
	if (contender->distance < results[results_length - 1].distance) {
       // replace the worst result with the new result
       results[results_length - 1] = *contender;

       // Sort the results array to keep the lowest distances in order
       for (
       	size_t j = results_length - 1;
       	j > 0 && results[j].distance < results[j - 1].distance;
       	j--
       ) {
           // Swap the results
           struct search_result_s temp = results[j];
           results[j] = results[j - 1];
           results[j - 1] = temp;
       }
   }
}
*/

static inline void search_insert_contender(
    struct search_result_s *results,
    size_t results_length,
    struct search_result_s *contender
) {
    // Only proceed if the contender is better than the worst result
    if (contender->distance < results[results_length - 1].distance) {
        // Perform a binary search to find the correct position for the new contender
        size_t low = 0;
        size_t high = results_length - 1;

        while (low < high) {
            size_t mid = (low + high) / 2;
            if (results[mid].distance > contender->distance) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        // Shift elements to the right to make room for the new contender
        for (size_t j = results_length - 1; j > low; j--) {
            results[j] = results[j - 1];
        }

        // Insert the new contender at the found position
        results[low] = *contender;
    }
}

#ifdef VECDB_USE_OPENMP
#include <omp.h>
#endif

void command_search(state_t *state, int argc, char*argv[]) {
    io_read_stdin(state);
        // read stdin into state->io_buffer

    io_parse_json_vector(state);
        // parse state->io_buffer as json into state->input_vector
    
    database_t db = io_map_database(state);

    const size_t results_length = 20;
    struct search_result_s global_results[results_length];

    // initialise results to infinity
    for(size_t i = 0; i < results_length; i++) {
        global_results[i].distance = INFINITY;
    }

	int start_time = clock();

	#ifdef VECDB_USE_OPENMP
	omp_set_num_threads(8);

	#pragma omp parallel
	#endif
	{
		// database_t db = io_map_database(state);

		#ifdef VECDB_USE_OPENMP
		int total_threads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		#else
		int total_threads = 1;
		int thread_id = 1;
		#endif

		struct search_result_s local_results[results_length];

		// initialise results to infinity
	    for(size_t i = 0; i < results_length; i++) {
	        local_results[i].distance = INFINITY;
	    }

		for(size_t i = thread_id; i < db.length; i += total_threads) {
			number_t hot_vector[state->dimensions];
	    
	        // copy vector from database into the aligned space for comparison
	        memcpy(
	            hot_vector,
	            database_get_chunk(state, &db, i),
	            get_vector_size(state)
	        );

	        
			struct search_result_s contender;
			
			contender.distance = get_distance(
	            state->dimensions,
	            state->input_vector,
	            hot_vector
	        );

	        contender.label = database_get_chunk_label(state, &db, i);

			search_insert_contender(
				local_results,
				results_length,
				&contender
			);

			if(i % 10000 == 0) {
				double cpu_time_used = ((double) (clock() - start_time)) / CLOCKS_PER_SEC;

				double mbs = ((i * get_vector_size(state)) / 1e+6) / cpu_time_used;
				
				fprintf(stderr, "\r%s searching at \033[4m%.2f\033[0m MB/s", vecdb, mbs);
			}
	    }


		#ifdef VECDB_USE_OPENMP
	    #pragma omp critical
	    #endif
		{
			for(size_t i = 0; i < results_length; i++) {
				search_insert_contender(
					global_results,
					results_length,
					&local_results[i]
				);
			}
		}
    }

	for(size_t i = 0; i < results_length; i++) {
		if(global_results[i].label != NULL) {
			printf("%s\n", global_results[i].label);
		}
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

    fprintf(stderr, "%s float type = fp%d\n", vecdb, sizeof(number_t) * 8);
    fprintf(stderr, "%s simd = %s\n", vecdb, VECDB_SIMD);
    fprintf(stderr, "%s state.path = '%s'\n", vecdb, state.path);
    fprintf(stderr, "%s state.dimensions = '%d'\n", vecdb, state.dimensions);
    fprintf(stderr, "%s state.label_size = '%d'\n", vecdb, state.label_size);
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
