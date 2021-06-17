import argparse
import pathlib
import shutil
import subprocess


def repair(wheel, dest_dir):
    cmd = "delvewheel.exe repair -w {dest_dir} {wheel}"
    cmd = cmd.format(wheel=wheel, dest_dir=dest_dir)
    out = subprocess.check_output(cmd, shell=True).decode()
    print("+ " + cmd)
    print("+ delvewheel.exe output:\n" + out)
    last_line = out.splitlines()[-1]

    # cibuildwheels expects the wheel to be in dest_dir but delvewheel does not copy the
    # wheel to dest_dir if there is nothing to repair
    if last_line.startswith("no external dependencies are needed"):
        print(f"+ Manually copying {wheel} to {dest_dir}")
        pathlib.Path(dest_dir).mkdir(exist_ok=True)
        shutil.copy2(wheel, dest_dir)
    else:
        print(f"+ No need for a manual copy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel")
    parser.add_argument("dest_dir")
    args = parser.parse_args()
    repair(args.wheel, args.dest_dir)
