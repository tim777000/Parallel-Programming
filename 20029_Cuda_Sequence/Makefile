all:
	gcc -std=c99 -O2 sequence.c -o sequence
	nvcc -Xcompiler '-O2 -fopenmp' main.cu -o main
clean:
	rm -f sequence
	rm -f main
