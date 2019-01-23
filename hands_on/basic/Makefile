ifndef GPI2_HOME
  GPI2_HOME=/opt/GPI2
endif

BIN += helloworld.exe
BIN += segments.exe
BIN += onesided.exe

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O3
# CFLAGS += -std=c99

CC = mpic++

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

default: $(BIN) jacobi_2d

%.exe: %.o $(addsuffix .o, $(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(addprefix -l, $(LIB))

jacobi_2d:
	$(CC) $(CFLAGS) -std=c++11 -o jacobi_2d.exe jacobi_2d.cxx ../aux/queue.c ../aux/waitsome.c $(LDFLAGS) $(addprefix -l, $(LIB))
	
###############################################################################

.PHONY: clean objclean

objclean:
	rm -f *.o

clean: objclean
	rm -f $(BIN)
