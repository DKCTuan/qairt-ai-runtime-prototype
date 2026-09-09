CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I.

.PHONY: test
test:
	$(CC) $(CFLAGS) traffic_p16.c tests/test_preprocessing.c -lm -o /tmp/traffic_p16_test
	/tmp/traffic_p16_test
