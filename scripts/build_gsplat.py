import os
import subprocess
import sys
import shutil
import torch
import stat
from pathlib import Path

# ============================================================================
# SAFETY CONFIGURATION
# ============================================================================
# "Consumer Product" Philosophy:
# 1. NEVER upgrade existing packages (torch, numpy) implicitly.
# 2. Build against the currently active environment.
# 3. Install ONLY the final binary.
# ============================================================================

def on_rm_error(func, path, exc_info):
    """Handle read-only files on Windows during cleanup."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def run_command(cmd, cwd=None, env=None, check=True):
    """Run command with detailed logging."""
    print(f"🚀 Running: {cmd}")
    sys.stdout.flush()
    try:
        if check:
            subprocess.check_call(cmd, shell=True, cwd=cwd, env=env)
            return 0
        else:
            return subprocess.call(cmd, shell=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error code {e.returncode}")
        if check:
            sys.exit(1)
        return e.returncode

def get_cuda_version():
    return torch.version.cuda

def check_compiler():
    nvcc_ok = False
    cl_ok = False
    try:
        subprocess.check_output("nvcc --version", shell=True, stderr=subprocess.STDOUT)
        nvcc_ok = True
    except: pass
    
    # Check for system cl.exe
    if shutil.which("cl.exe"):
        cl_ok = True
    else:
        # Fallback check
        try:
            subprocess.check_output("cl", shell=True, stderr=subprocess.STDOUT)
            cl_ok = True
        except: pass
        
    return nvcc_ok, cl_ok

def build_gsplat():
    print("\n==================================================")
    print("   �️  Safe gsplat Installer (Surgical Mode)")
    print("==================================================")

    # 1. Audit Environment (Read-Only)
    print(f"✅ Python: {sys.version.split()[0]}")
    print(f"✅ PyTorch: {torch.__version__}")
    
    cuda_ver = get_cuda_version()
    if not cuda_ver:
        print("❌ CUDA not found! gsplat requires a CUDA-enabled PyTorch.")
        sys.exit(1)
    print(f"✅ PyTorch CUDA: {cuda_ver}")

    nvcc_ok, cl_ok = check_compiler()
    
    script_dir = Path(__file__).parent.resolve()
    msvc_dir = script_dir / "portable_msvc"
    use_portable_msvc = False
    
    # Detect Portable Compiler availability
    if not cl_ok:
        if (msvc_dir / "MSVC").exists() or (msvc_dir / "MSVC-Portable.bat").exists():
            use_portable_msvc = True
    
    print(f"🔍 Compass Check:")
    print(f"   - NVCC: {'✅ Found' if nvcc_ok else '❌ MISSING'}")
    print(f"   - CL:   {'✅ Found (System)' if cl_ok else ('✅ Found (Portable)' if use_portable_msvc else '❌ MISSING')}")
    
    if not nvcc_ok:
        print("\n❌ CRITICAL: NVCC (CUDA Compiler) is missing.")
        print("Please install the CUDA Toolkit that matches your PyTorch version.")
        print("This script will NOT attempt to install system-level drivers.")
        sys.exit(1)
        
    if not cl_ok and not use_portable_msvc:
         print("\n⚠️  MSVC Compiler missing. Attempting to download Portable MSVC (600MB)...")
         # This is the ONLY external modification we allow (downloading a tool to a subfolder)
         try:
             run_command(f"git clone https://github.com/Delphier/MSVC {msvc_dir}")
             use_portable_msvc = True
         except:
             print("❌ Failed to download compiler. Please install Visual Studio Build Tools manually.")
             sys.exit(1)

    # 2. Check for Existing Wheel (Fast Path)
    # We attempt to fetch a wheel, but we do NOT assume we can verify its compatibility fully automatically
    # so we rely on pip's logic, BUT we restrict implicit upgrades.
    print("\n🛞  Checking for pre-built wheel...")
    try:
        pt_tag = f"pt{torch.__version__.split('+')[0].replace('.', '')[:2]}"
        cu_tag = f"cu{cuda_ver.replace('.', '')}"
        index_url = f"https://docs.gsplat.studio/whl/{pt_tag}{cu_tag}"
        
        # We try install with --no-deps immediately. If it works, great.
        cmd = f"{sys.executable} -m pip install gsplat --index-url {index_url} --no-deps"
        if run_command(cmd, check=False) == 0:
            print("✅ Installed from official wheel!")
            verify_install()
            return
    except:
        pass
    
    print("🔧 Official wheel not found or incompatible. Proceeding to Safe Build...")

    # 3. Clone Source (Local Cache)
    build_dir = script_dir / "gsplat_build"
    if not build_dir.exists():
        print(f"📥 Cloning gsplat source...")
        run_command(f"git clone --recursive https://github.com/nerfstudio-project/gsplat.git {build_dir}")
    else:
        print("📂 Source cache found.")

    # 4. Build Wheel
    # We build a wheel instead of direct install to separate compilation from installation
    print("\n🏗️  Compiling gsplat...")
    dist_dir = script_dir / "dist"
    dist_dir.mkdir(exist_ok=True)
    
    env = os.environ.copy()
    
    # Check for ninja (speed up build)
    try: import ninja
    except: 
        print("Installing ninja build tool...")
        run_command(f"{sys.executable} -m pip install ninja")

    # The Build Command
    # --no-build-isolation: Uses YOUR torch, YOUR numpy. Does not create a venv.
    wheel_cmd = f"{sys.executable} -m pip wheel . -w {dist_dir} --verbose --no-build-isolation"

    if use_portable_msvc:
        # Wrapper logic for portable compiler
        msvc_installed = msvc_dir / "MSVC"
        vcvars = list(msvc_installed.rglob("vcvars64.bat"))
        
        # Determine activator
        activator = None
        if vcvars: 
            activator = vcvars[0]
        else:
             # Run setup if needed
             print("⏳ Initializing Portable MSVC...")
             subprocess.check_call(f'"{msvc_dir}/MSVC-Portable.bat"', shell=True, cwd=str(msvc_dir))
             vcvars = list(msvc_installed.rglob("vcvars64.bat"))
             if vcvars: activator = vcvars[0]
        
        if activator:
             full_cmd = f'"{activator}" && {wheel_cmd}'
        else:
             print("❌ Compiler setup failed.")
             sys.exit(1)
    else:
        full_cmd = wheel_cmd

    # Execute Build
    run_command(full_cmd, cwd=str(build_dir), env=env)
    
    # 5. Surgical Install
    # Find the wheel we just made
    try:
        whl = sorted(list(dist_dir.glob("*.whl")), key=os.path.getmtime)[-1]
    except IndexError:
        print("❌ Build failed to produce a .whl file.")
        sys.exit(1)

    print(f"\n📦 Safe Installation of {whl.name}...")
    
    # CRITICAL: --no-deps
    # This guarantees we DO NOT TOUCH numpy, torch, or anything else.
    # We only copy the gsplat files.
    install_cmd = f"{sys.executable} -m pip install {whl} --force-reinstall --no-deps"
    run_command(install_cmd)
    
    print("\n==================================================")
    print("🎉 SUCCESS")
    print("==================================================")
    verify_install()

def verify_install():
    try:
        import gsplat
        print(f"✅ gsplat {gsplat.__version__} is importable.")
    except Exception as e:
        print(f"⚠️ Installed but import failed: {e}")

if __name__ == "__main__":
    build_gsplat()
