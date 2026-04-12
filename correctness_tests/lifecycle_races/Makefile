NVCC      = nvcc
NVCCFLAGS = -lpthread

SRCS = $(wildcard tc*.cu)
BINS = $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
