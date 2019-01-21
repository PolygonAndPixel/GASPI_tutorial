ifndef GPI2_HOME
  GPI2_HOME=/opt/GPI2
endif

BIN += helloworld.exe
BIN += segments.exe
BIN += onesided.exe
BIN += 1d_jacobi.exe
BIN += 2d_jacobi.exe

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O3
CFLAGS += -std=c99

CC = mpicc

###############################################################################

INCLUDE_DIR += $(GPI2_HOME)/include
INCLUDE_DIR += ..
LIBRARY_DIR += $(GPI2_HOME)/lib64

LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))

CFLAGS += $(addprefix -I,$(INCLUDE_DIR))

LIB += GPI2
LIB += ibverbs
LIB += pthread
LIB += m

OBJ += ../aux/queue
OBJ += ../aux/waitsome

###############################################################################

default: $(BIN)

%.exe: %.o $(addsuffix .o, $(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(addprefix -l, $(LIB))

###############################################################################

.PHONY: clean objclean

objclean:
	rm -f *.o

clean: objclean
	rm -f $(BIN)
