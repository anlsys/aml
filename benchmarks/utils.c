#include "utils.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static const char *namespace;
static int active = 0;

void log_init(const char *nm) {
	char *debug = getenv("DEBUG");
	if(debug)
		active = atoi(debug);
}

void log_msg(const char *level, unsigned int line, const char *fmt, ...)
{
	va_list ap;
	if(!active)
		return;
	printf("%s:\t%s:\t%u:\t", namespace, level, line);
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
}
