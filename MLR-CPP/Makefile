files := *.cpp *.h
headers := ./external

main : $(files)
	g++ $(files) -I $(headers) -o main

.PHONY: clean
clean :
	rm -rf ./main
