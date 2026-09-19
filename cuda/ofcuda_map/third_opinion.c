/* Third opinion for the 100-map terrain sweep.
 *
 * Deliberately shares nothing with the Rust side: own argv handling, own dims
 * (passed in from the manifest by the shell), own FNV-1a 64, own land-bit test.
 * Reads the raw plane file, hashes it in row-major file order (index y*w+x is
 * the file order for every map in resources/maps), counts bit 0x80.
 *
 * usage: map_c_fnv <file> <width> <height>
 * prints: <width>x<height>\t<land>\t<0xhash>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <file> <width> <height>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    unsigned long w = strtoul(argv[2], NULL, 10);
    unsigned long h = strtoul(argv[3], NULL, 10);
    unsigned long n = w * h;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "open %s: failed\n", path);
        return 1;
    }
    unsigned char *buf = malloc(n ? n : 1);
    if (!buf) {
        fprintf(stderr, "oom\n");
        return 1;
    }
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "%s: short read (wanted %lu bytes)\n", path, n);
        return 1;
    }
    fclose(f);

    uint64_t hash = 0xcbf29ce484222325ULL;
    unsigned long land = 0;
    /* Row-major walk: index = y*w + x, exactly the order the bytes sit in. */
    for (unsigned long y = 0; y < h; y++) {
        for (unsigned long x = 0; x < w; x++) {
            unsigned char b = buf[y * w + x];
            hash = (hash ^ (uint64_t)b) * 0x100000001b3ULL;
            if (b & 0x80u) {
                land++;
            }
        }
    }
    printf("%lux%lu\t%lu\t0x%016llx\n", w, h, land, (unsigned long long)hash);
    free(buf);
    return 0;
}
