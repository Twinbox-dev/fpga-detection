TARGET = plite-test
CFLAGS = -O3 -Wall -std=c++11 `pkg-config --cflags opencv`
CFLAGS += -march=armv7-a -mfloat-abi=hard -mfpu=neon

LDFLAGS = -O3 -Wall -std=c++11 `pkg-config --libs opencv`

CC = g++

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CC) $(LDFLAGS) -o $@ $^ `pkg-config --libs opencv` -lpthread

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) *.a *.o *~