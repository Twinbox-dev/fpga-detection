import time
from Arm_Lib import Arm_Device

# 初始化机械臂
Arm = Arm_Device()
time.sleep(0.1)  # 等待设备就绪

print("移动 1 号舵机到 90°")
Arm.Arm_serial_servo_write(1, 90, 1000)  # 1000ms 内到达
time.sleep(1.5)

print("移动 1 号舵机到 0°")
Arm.Arm_serial_servo_write(1, 0, 1000)
time.sleep(1.5)

print("移动 1 号舵机回到 90°")
Arm.Arm_serial_servo_write(1, 90, 1000)
time.sleep(1.5)

print("测试完成。")