NVCC      = nvcc
NVCCFLAGS = -O2 -lm

SRCS = tc02_performance_overhead.cu tc03_random_write_overhead.cu
BINS = $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
