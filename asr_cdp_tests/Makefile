.PHONY: all clean

CC=gcc
CFLAGS=-c -Wall
LDFLAGS=-lm

all: asr_cdp_tests

asr_cdp_tests : AllTests.o CuTest.o json.o asr_cdp_lib.o
	$(CC) AllTests.o CuTest.o json.o asr_cdp_lib.o -o asr_cdp_tests $(LDFLAGS)

AllTests.o : AllTests.c CuTest.h ../asr_cdp/asr_cdp_lib.h
	$(CC) $(CFLAGS) AllTests.c

CuTest.o : CuTest.c CuTest.h
	$(CC) $(CFLAGS) CuTest.c

json.o : ../asr_cdp/json.c ../asr_cdp/json.h
	$(CC) $(CFLAGS) ../asr_cdp/json.c

asr_cdp_lib.o : ../asr_cdp/asr_cdp_lib.c ../asr_cdp/asr_cdp_lib.h ../asr_cdp/json.h
	$(CC) $(CFLAGS) ../asr_cdp/asr_cdp_lib.c

clean :
	rm -rf *.o asr_cdp_tests
