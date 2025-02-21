# Target rules
all: build

build: add

OBJ =  add.o tool.o public.o
INCLUDES := -I${NEUWARE_HOME}/include
LIBRARIES := -L${NEUWARE_HOME}/lib64
CXXFLAGS := -Og -std=c++11 -fPIC -lstdc++ -Wall -pthread
LDFLAGS := -lcnnl -lcnrt -lcndrv

add: $(OBJ)
	${CXX} -o $@ $+ $(LIBRARIES) $(LDFLAGS)

%.o: %.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $^
tool.o: ../public/tool.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $^
public.o : ../public/public.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

clean:
	rm -rf *o
	rm -rf add

clobber: clean
