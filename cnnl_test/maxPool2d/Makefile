# Target rules
all: build

build: maxPool2d

OBJ =  maxPool2d.o tool.o public.o
INCLUDES := -I${NEUWARE_HOME}/include
LIBRARIES := -L${NEUWARE_HOME}/lib64
CXXFLAGS := -Og -std=c++11 -fPIC -lstdc++ -Wall -O3 -pthread
LDFLAGS := -lcnnl -lcnrt -lcndrv

maxPool2d: $(OBJ)
	${CXX} -o $@ $+ $(LIBRARIES) $(LDFLAGS)

%.o: %.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $^
tool.o: ../public/tool.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $^
public.o : ../public/public.cc
	${CXX} $(INCLUDES) $(CXXFLAGS) -o $@ -c $^

clean:
	rm -rf *o
	rm -rf maxPool2d

clobber: clean
