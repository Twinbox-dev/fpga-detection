#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>

// 使用说明
void print_usage(const char *prog_name) {
    printf("用法: %s [选项]\n", prog_name);
    printf("选项:\n");
    printf("  -w <值>    写入寄存器值（十六进制，如 0xFF）\n");
    printf("  -r         读取寄存器值\n");
    printf("  -a <地址>  指定寄存器地址偏移（十六进制，默认 0x10）\n");
    printf("  -c <次数>  循环次数（与 -w 一起使用）\n");
    printf("  -d <秒>    延时秒数（默认 1 秒）\n");
    printf("  -h         显示此帮助信息\n");
    printf("\n示例:\n");
    printf("  %s -r                # 读取寄存器值\n", prog_name);
    printf("  %s -w 0x55           # 写入 0x55\n", prog_name);
    printf("  %s -w 0xAA -c 5 -d 2 # 写入 0xAA，循环5次，每次延时2秒\n", prog_name);
    printf("  %s -r -a 0x20        # 读取偏移 0x20 处的值\n", prog_name);
}

int main(int argc, char *argv[]) {
    int fd;
    volatile unsigned int *reg_addr;
    void *map_base;
    
    // 默认参数
    unsigned int write_val = 0;
    unsigned int reg_offset = 0x10;  // 默认偏移
    int do_write = 0;
    int do_read = 0;
    int loop_count = 1;
    int delay_sec = 1;
    
    // 解析命令行参数
    int opt;
    while ((opt = getopt(argc, argv, "w:ra:c:d:h")) != -1) {
        switch (opt) {
            case 'w':
                write_val = strtoul(optarg, NULL, 0);
                do_write = 1;
                break;
            case 'r':
                do_read = 1;
                break;
            case 'a':
                reg_offset = strtoul(optarg, NULL, 0);
                if (reg_offset >= 0x1000) {
                    fprintf(stderr, "错误: 偏移地址超出范围 (0-0xFFF)\n");
                    return -1;
                }
                break;
            case 'c':
                loop_count = atoi(optarg);
                if (loop_count <= 0) loop_count = 1;
                break;
            case 'd':
                delay_sec = atoi(optarg);
                if (delay_sec <= 0) delay_sec = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                fprintf(stderr, "使用 -h 查看帮助信息\n");
                return -1;
        }
    }
    
    // 如果没有指定任何操作
    if (!do_write && !do_read) {
        printf("警告: 未指定操作，默认执行读取操作\n");
        do_read = 1;
    }
    
    // 打开 /dev/mem
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "打开 /dev/mem 失败: %s\n", strerror(errno));
        return -1;
    }
    
    // 映射 FPGA 地址空间
    map_base = mmap(NULL, 0x1000, 
                   PROT_READ | PROT_WRITE, 
                   MAP_SHARED, fd, 0xC0000000);
    
    if (map_base == MAP_FAILED) {
        fprintf(stderr, "内存映射失败: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    
    // 计算寄存器地址
    reg_addr = (volatile unsigned int *)((char *)map_base + reg_offset);
    
    printf("寄存器地址: 0xC0000000 + 0x%X = 0xC000%04X\n", 
           reg_offset, reg_offset);
    
    // 执行写入操作
    if (do_write) {
        printf("写入值: 0x%08X, 循环次数: %d, 延时: %d秒\n", 
               write_val, loop_count, delay_sec);
        
        for (int i = 0; i < loop_count; i++) {
            *reg_addr = write_val;
            printf("[%d/%d] 写入: 0x%08X", i+1, loop_count, write_val);
            
            if (do_read) {
                unsigned int read_val = *reg_addr;
                printf(" -> 读取: 0x%08X", read_val);
            }
            printf("\n");
            
            if (i < loop_count - 1) {
                sleep(delay_sec);
            }
        }
    }
    
    // 只执行读取操作
    else if (do_read) {
        unsigned int read_val = *reg_addr;
        printf("读取结果: 0x%08X (%u)\n", read_val, read_val);
    }
    
    // 清理
    munmap(map_base, 0x1000);
    close(fd);
    
    return 0;
}