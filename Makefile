CXX = g++ -std=c++14 -g
CPPFLAGS =  `pkg-config --cflags opencv`
LDLIBS = `pkg-config --libs opencv` -Iinclude/

SRCS = $(wildcard src/*.cpp *.cpp)

OBJS = $(SRCS:.cpp=.o)


all: $(OBJS)
	$(CXX) $(CPPFLAGS) $(OBJS) $(LDLIBS) -o main.out

$(OBJS): %.o: %.cpp
	$(CXX) -c $(CPPFLAGS) $< $(LDLIBS) -o $@


clean:
	rm -f $(OBJS)
