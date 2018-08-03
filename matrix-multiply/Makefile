CXX := nvcc
TARGET := benchmark
LIBS := -L /opt/cuda-8.0/lib64
CXXFLAGS := -arch=sm_35 -std=c++11 -O2

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcublas

.phony: clean

clean:
	rm $(TARGET) || echo -n ""

