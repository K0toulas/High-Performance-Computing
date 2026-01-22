# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= test1 test2 test3 test4 test5 test6 test7 test8 test9 test10 test11 test12 test13
 

#This is the compiler to use
CC = icx

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -O0

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm


# make all will create all executables
all: $(EXECUTABLES)

# This is the rule to create any executable from the corresponding .c 
# file with the same name.
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# make clean will remove all executables, jpg files and the 
# output of previous executions.
clean:
	rm -f $(EXECUTABLES) *.jpg output_sobel.grey

# make image will create the output_sobel.jpg from the output_sobel.grey. 
# Remember to change this rule if you change the name of the output file.
image: output_sobel.grey
	convert -depth 8 -size 4096x4096 GRAY:output_sobel.grey output_sobel.jpg 

