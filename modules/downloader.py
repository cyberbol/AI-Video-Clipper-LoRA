#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# AI Video Clipper & LoRA Captioner - High Performance Downloader
# Contributor: FNGarvin | License: MIT
# --------------------------------------------------------------------------------

import os
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import sys
import signal
from tqdm import tqdm
import ctypes
try:
    from ctypes import wintypes
except ImportError:
    wintypes = None

# --- ABSOLUTE NUCLEAR WINDOWS SHUTDOWN (API LEVEL) ---
if os.name == 'nt' and wintypes:
    # Use SetConsoleCtrlHandler to catch Ctrl+C even in worker threads
    PHANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    
    def windows_ctrl_handler(ctrl_type):
        # 0 is CTRL_C_EVENT, 1 is CTRL_BREAK_EVENT, 2 is CTRL_CLOSE_EVENT
        if ctrl_type in (0, 1, 2):
            sys.stdout.write("\n🛑 NUCLEAR SHUTDOWN: TERMINATING IMMEDIATELY...\n")
            sys.stdout.flush()
            # Absolute hard kill
            ctypes.windll.kernel32.ExitProcess(1)
            return True
        return False

    # Keep reference to preventing GC
    _handler_ref = PHANDLER_ROUTINE(windows_ctrl_handler)
    if not ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler_ref, True):
        print("⚠️ Warning: Could not register Windows nuclear handler.")


# Configuration
CHUNK_SIZE = 1024 * 1024  # 1MB buffer for streaming
MULTI_CONN_THRESHOLD = 50 * 1024 * 1024  # 50MB: Files larger than this use multi-connection
CONNS_PER_FILE = 8  # Simultaneous connections for a single large file
MAX_PARALLEL_FILES = 4  # How many files to start at once

def list_repo_files(model_id):
    """
    Fetches the list of files from a Hugging Face model repository.
    Bypasses hf_hub dependency/offline checks by using the HF API directly.
    """
    url = f"https://huggingface.co/api/models/{model_id}/tree/main?recursive=true"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [item["path"] for item in data if item.get("type") == "file"]
    except Exception as e:
        raise RuntimeError(f"Failed to list repo files for {model_id}: {e}")

class ModelDownloader:
    """
    Manages the download of files from a Hugging Face model repository.
    Supports multi-connection downloads for large files and parallel file downloads.
    """
    def __init__(self, model_id, target_base_dir, specific_files=None, log_callback=None):
        self.model_id = model_id
        self.dest_path = os.path.normpath(os.path.join(target_base_dir, model_id))
        self.specific_files = specific_files
        self.log_callback = log_callback
        self.shutdown_event = threading.Event()
        self.threads = [] # Tracks spawned threads
        self.error = None
        self.is_running = False
        self.is_cancelled = False
        self.is_completed = False
        self.total_files_to_download = 0
        self.files_downloaded_count = 0
        self.current_file_progress = {}

    def log(self, msg):
        """Logs a message to console and an optional callback."""
        print(msg)
        if self.log_callback:
            try: self.log_callback(msg)
            except: pass

    def _download_single_file(self, url, local_path):
        """
        Downloads a single file, supporting retries and multi-connection for large files.
        """
        if self.shutdown_event.is_set(): return False
        
        part_path = local_path + ".part"
        filename = os.path.basename(local_path)
        
        for attempt in range(3):
            try:
                if self.shutdown_event.is_set(): return False
                
                # Check headers to get total size and range support
                try:
                    res = requests.head(url, allow_redirects=True, timeout=10)
                    total_size = int(res.headers.get("content-length", 0))
                    accept_ranges = res.headers.get("accept-ranges") == "bytes"
                except:
                    # Fallback if HEAD fails
                    total_size = 0
                    accept_ranges = False

                if total_size == 0:
                    res = requests.get(url, stream=True, timeout=10)
                    total_size = int(res.headers.get("content-length", 0))

                from tqdm import tqdm
                
                # --- MULTI-THREADED DOWNLOAD ---
                if accept_ranges and total_size > MULTI_CONN_THRESHOLD:
                    if attempt == 0:
                        self.log(f"   🚀 Large file: {filename} ({total_size/(1024*1024):.1f}MB). Using {CONNS_PER_FILE} conns.")
                    else:
                        self.log(f"   🔄 Retry {attempt+1}/3: {filename}")
                    
                    with open(part_path, "wb") as f:
                        f.truncate(total_size)
                        
                    downloaded = 0
                    lock = threading.Lock()
                    
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"   {filename}", file=sys.stdout, dynamic_ncols=True) as pbar:
                        def update_progress(bytes_len):
                            nonlocal downloaded
                            with lock:
                                downloaded += bytes_len
                                pbar.update(bytes_len)
                                if total_size > 0:
                                    self.current_file_progress[filename] = int((downloaded / total_size) * 100)

                        def segment_worker(start, end, result_list, index):
                            try:
                                headers = {"Range": f"bytes={start}-{end}"}
                                with requests.get(url, headers=headers, stream=True, timeout=20) as r:
                                    r.raise_for_status()
                                    with open(part_path, "r+b") as f:
                                        f.seek(start)
                                        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                                            if self.shutdown_event.is_set(): return
                                            if chunk:
                                                f.write(chunk)
                                                update_progress(len(chunk))
                                result_list[index] = True
                            except Exception:
                                result_list[index] = False

                        seg_threads = []
                        seg_results = [False] * CONNS_PER_FILE
                        seg_size = total_size // CONNS_PER_FILE

                        for i in range(CONNS_PER_FILE):
                            start = i * seg_size
                            end = start + seg_size - 1 if i < CONNS_PER_FILE - 1 else total_size - 1
                            t = threading.Thread(target=segment_worker, args=(start, end, seg_results, i))
                            t.daemon = True
                            t.start()
                            seg_threads.append(t)
                            self.threads.append(t)

                        for t in seg_threads:
                            while t.is_alive():
                                t.join(timeout=0.05) # Short timeout to check shutdown_event
                                if self.shutdown_event.is_set(): break
                        
                        if self.shutdown_event.is_set(): 
                            return False
                        if not all(seg_results): raise RuntimeError("Segment download failed")

                # --- SINGLE THREADED DOWNLOAD ---
                else:
                    if total_size > 1024*1024: 
                        if attempt == 0: self.log(f"   📥 Downloading {filename} ({total_size/(1024*1024):.1f}MB)...")
                        else: self.log(f"   🔄 Retry {attempt+1}/3: {filename}")
                        
                    response = requests.get(url, stream=True, timeout=20)
                    response.raise_for_status()
                    
                    downloaded = 0
                    with open(part_path, "wb") as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"   {filename}", file=sys.stdout, dynamic_ncols=True) as pbar:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if self.shutdown_event.is_set(): return False
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                                    downloaded += len(chunk)
                                    if total_size > 0:
                                        self.current_file_progress[filename] = int(downloaded/total_size * 100)

                # Atomic Rename (if not stopped)
                if self.shutdown_event.is_set():
                    if os.path.exists(part_path):
                        try: os.remove(part_path)
                        except: pass
                    return False
                    
                if os.path.exists(local_path): os.remove(local_path)
                os.rename(part_path, local_path)
                self.current_file_progress[filename] = 100
                return True # Success, exit loop

            except Exception as e:
                self.log(f"   ⚠️ Attempt {attempt+1} failed for {filename}: {e}")
                if self.shutdown_event.is_set(): return False
                time.sleep(2) # Backoff
        
        self.log(f"   ❌ All 3 attempts failed for {filename}")
        if os.path.exists(part_path):
            try: os.remove(part_path)
            except: pass
        return False


    def run(self):
        """
        Initiates the download process for the model.
        Checks for existing files, queues downloads, and manages worker threads.
        """
        self.is_running = True
        os.makedirs(self.dest_path, exist_ok=True)
        
        # Offline-First: Check if files already exist
        if self.specific_files and all(os.path.exists(os.path.join(self.dest_path, f)) for f in self.specific_files):
            self.log(f"✅ Model {self.model_id} found locally.")
            self.is_completed = True
            self.is_running = False
            return self.dest_path

        # Secondary Check: Complete Flag (for full repo downloads)
        complete_flag = os.path.join(self.dest_path, ".download_complete")
        if os.path.exists(complete_flag):
            if not self.specific_files: 
                 self.log(f"✅ Model {self.model_id} up-to-date (Flag).")
                 self.is_completed = True
                 self.is_running = False
                 return self.dest_path
            else:
                 # If specific files are requested, verify their existence despite the flag
                 if all(os.path.exists(os.path.join(self.dest_path, f)) for f in self.specific_files):
                     self.log(f"✅ Model {self.model_id} verified locally.")
                     self.is_completed = True
                     self.is_running = False
                     return self.dest_path

        self.log(f"🔎 Validating / Downloading: {self.model_id}\n   (Check console log for progress bars)")
        
        try:
            # 1. Fetch List of files from repo
            available_files = list_repo_files(self.model_id)
            
            if self.specific_files:
                files_to_download = [f for f in available_files if f in self.specific_files]
            else:
                exclude = [".git", ".gitattributes", ".download_complete"]
                files_to_download = [f for f in available_files if not any(x in f for x in exclude)]

            self.total_files_to_download = len(files_to_download)
            
            # 2. Queue files for download
            q = queue.Queue()
            for f in files_to_download: q.put(f)

            results_lock = threading.Lock()
            
            def worker():
                while not self.shutdown_event.is_set():
                    try: 
                        f_name = q.get_nowait()
                    except queue.Empty: 
                        break # No more files in queue
                    
                    try:
                        file_url = f"https://huggingface.co/{self.model_id}/resolve/main/{f_name}?download=true"
                        local = os.path.normpath(os.path.join(self.dest_path, f_name))
                        os.makedirs(os.path.dirname(local), exist_ok=True)
                        
                        if os.path.exists(local):
                            with results_lock:
                                self.files_downloaded_count += 1
                            self.log(f"   ✅ Verified: {f_name} ({self.files_downloaded_count}/{self.total_files_to_download})")
                        else:
                            # Start fresh: Remove any stale .part files
                            part_file = local + ".part"
                            if os.path.exists(part_file):
                                try: os.remove(part_file)
                                except: pass
                                
                            if self._download_single_file(file_url, local):
                                with results_lock:
                                    self.files_downloaded_count += 1
                                self.log(f"   ✅ Downloaded: {f_name} ({self.files_downloaded_count}/{self.total_files_to_download})")
                            else:
                                self.log(f"   ⚠️ Warning: {f_name} failed or cancelled.")
                                self.current_file_progress[f_name] = -1 
                                if self.shutdown_event.is_set():
                                    break 
                    except Exception as e:
                        self.log(f"   ❌ Error processing {f_name}: {e}")
                        self.current_file_progress[f_name] = -1
                    finally:
                        q.task_done()
            
            # Start Workers
            workers = []
            for _ in range(min(MAX_PARALLEL_FILES, self.total_files_to_download)):
                t = threading.Thread(target=worker)
                t.daemon = True
                t.start()
                workers.append(t)
                self.threads.append(t) # Track for kill

            # Wait for threads, allowing Ctrl+C / shutdown_event
            for t in workers:
                while t.is_alive():
                    t.join(timeout=0.1) # Short timeout allows checking for shutdown_event
                    if self.shutdown_event.is_set():
                        break # Eject from waiting for this thread

            # Ensure all queue tasks are marked done, even if cancelled
            if self.shutdown_event.is_set():
                # Drain the queue to allow workers to exit
                try:
                    while not q.empty():
                        q.get_nowait()
                        q.task_done()
                except queue.Empty: pass
            
            q.join() 
            
            if self.shutdown_event.is_set():
                self.is_cancelled = True
                if os.name == 'nt':
                    os._exit(1) 
                else:
                    sys.exit(1)
                
            if self.files_downloaded_count == self.total_files_to_download:
                with open(complete_flag, "w") as f:
                    f.write(f"verified_{time.time()}")
                self.log(f"✨ Model {self.model_id} verified 100%.")
                self.is_completed = True
                return self.dest_path
            else:
                raise RuntimeError(f"Download incomplete: {self.files_downloaded_count}/{self.total_files_to_download} files downloaded.")

        except KeyboardInterrupt:
            if os.name == 'nt':
                os._exit(1)
            self.cancel()
            self.error = "Download cancelled by user."
            raise  
        except Exception as e:
            self.error = e
            self.log(f"❌ Transfer Failed: {e}")
            raise e
        finally:
            self.is_running = False
            # Ensure all threads are signaled to stop
            self.shutdown_event.set()

    def cancel(self):
        """Signals all download operations to stop."""
        if not self.is_running and not self.is_completed:
            self.log("Download not running or already completed/cancelled.")
            return
        self.log("\n🛑 ABORTING DOWNLOAD...")
        self.shutdown_event.set()
        self.is_cancelled = True

# Compatibility Wrapper
def download_model(model_id, target_base_dir, specific_files=None, log_callback=None):
    """
    Convenience function to download a Hugging Face model.
    """
    downloader = ModelDownloader(model_id, target_base_dir, specific_files, log_callback)
    return downloader.run()

# EOF downloader.py
