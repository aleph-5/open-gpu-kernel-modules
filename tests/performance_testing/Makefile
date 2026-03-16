NVCC      = nvcc
NVCCFLAGS = -O2 -lm

SRCS = tc01_stress_testing.cu tc02_performance_overhead.cu
BINS = $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean

