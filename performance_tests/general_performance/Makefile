NVCC      = nvcc
NVCCFLAGS = -O2 -lm

SRCS = $(wildcard tc*.cu)
BINS = $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
