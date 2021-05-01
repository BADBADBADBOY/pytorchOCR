CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)

DEPS = $(shell find include -xtype f)
CXX_SOURCES = cppdbprocess.cpp include/clipper/clipper.cpp
OPENCV = `pkg-config --cflags --libs opencv`

LIB_SO = cppdbprocess.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)


