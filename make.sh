# make nms 
cd ptocr/postprocess/lanms/
make clean
make

# make pan,pse
cd ../piexlmerge
make clean
make

# make dbprocess
cd ../dbprocess
make clean
make