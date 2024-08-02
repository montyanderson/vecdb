# vecdb

> Toy vector database written in c99.

## features

* 🚀 Fast
    - Uses memory-mapped io to take advantage of your operating system's file cache
    - Supports ARM Neon SIMD (for M1, M2, etc)
    - Almost no syscalls
* 🤏 Low Memory Usage
    - Outside of `mmap()`, under two megabytes of heap is allocated
    - `malloc()` only called once at startup
* 🔗 Model Agnostic
    - Takes any embeddings as JSON into `stdin`
    - `discogs-effnet` adapter included
* 🧱 Built to last
    - No dependencies
    - Single file of c99 (plus a header-only json parser)

## usage

Build it. Requires a C compiler.

``` shell
./build.sh
```

Tell it:

* Where your database should go
* How many dimensions your vectors have
    - Must be a multiple of 4
* What amount of space to keep for text labels/ids

``` shell
export VECDB_PATH=$PWD/db.vec
export VECDB_DIMENSIONS=1280
export VECDB_LABEL_SIZE=1024
```

Add an embedding.

```
./adaptors/discogs-effnet/generate my-track.mp3 | ./vecdb add "my-track.mp3"
```

List.

```
./vecdb list
```

Search.

```
./adaptors/discogs-effnet/generate another-track.mp3 | ./vecdb add "another-track.mp3"
```

More instructions coming soon.

## todo

* [x] basic functionality
* [x] arm neon simd
* [ ] x86 simd
* [ ] benchmark
* [ ] pretty output
* [ ] n-ranked search results