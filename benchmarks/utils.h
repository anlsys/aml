#ifndef AML_BENCHS_UTILS_H
#define AML_BENCHS_UTILS_H 1

void log_init(const char *nm);
void log_msg(const char *level, unsigned int line, const char *fmt, ...);

#define debug(...) log_msg("debug", __LINE__, __VA_ARGS__)

#endif // AML_BENCHS_UTILS_H
