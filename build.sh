#!/bin/sh
gcc \
    --std=c99 \
    -Wall \
    -Wextra \
    -O3 \
    vecdb.c \
    -S \
    -fverbose-asm \
    -o vecdb.s

gcc \
    --std=c99 \
    -Wall \
    -Wextra \
    -O3 \
    vecdb.c \
    -o vecdb