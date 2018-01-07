.PHONY: all clean

CC=gcc
CFLAGS=-c -Wall -m64 -Ofast -flto -march=native -funroll-loops
LDFLAGS=-lm

all: asr_cdp

asr_cdp : main.o json.o asr_cdp_lib.o
	$(CC) main.o json.o asr_cdp_lib.o -o asr_cdp $(LDFLAGS)

main.o : main.c asr_cdp_lib.h
	$(CC) $(CFLAGS) main.c

json.o : json.c json.h
	$(CC) $(CFLAGS) json.c

asr_cdp_lib.o : asr_cdp_lib.c asr_cdp_lib.h json.h
	$(CC) $(CFLAGS) asr_cdp_lib.c

clean :
	rm -rf *.o asr_cdp
