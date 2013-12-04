# ExampleTests Project
SRCS = main.cpp
HDRS = 
PROJ = aca

# Remaining lines shouldn't need changing
# Here's what they do:
#   - rebuild if any header file or this Makefile changes
#   - include CppUnit as dynamic library
#   - search /opt/local for MacPorts
#   - generate .exe files for Windows
#   - add -enable-auto-import flag for Cygwin only

CC = g++
OBJS = $(SRCS:.cpp=.o)
APP = $(PROJ)
CFLAGS = -c -g -Wall `pkg-config --cflags opencv`
LIBS = -lUnitTest++ `pkg-config --libs opencv`

# gcc 4 in Cygwin needs non-standard --enable-auto-import option
ifneq (,$(findstring CYGWIN,$(shell uname)))
  LDFLAGS += -Wl,--enable-auto-import
endif

all: $(APP)

$(APP): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) $(LIBS) -o $(APP)

%.o: %.cpp $(HDRS)
	$(CC) $(CFLAGS) $< -o $@
clean:
	rm -f *.o $(APP)


