#!/bin/sh

export flags="--std=c99 -Wall -Wextra -march=native -mtune=native -O3 -g"
export source=vecdb.c
export asm=vecdb.s

# build asm output
cc \
	$flags \
	$source \
    -S \
    -fverbose-asm \
    -o $asm

# build executable
cc \
	$flags \
	$asm \
    -o vecdb
