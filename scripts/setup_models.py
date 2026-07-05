import os
import subprocess
import glob


def setup_model(modelfile_path):
    """
    Sets up an LLM model from Huggingface based on a Modelfile.
    """
    model_name = os.path.splitext(os.path.basename(modelfile_path))[0].lower()

    print(f"Creating model '{model_name}' from file...")
    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            check=True
        )
        print(f"Successfully created {model_name}.")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to create {model_name}")
        return False


def main():
    modelfiles = glob.glob("models/*.Modelfile")

    if not modelfiles:
        print("No .Modelfile found in models/ folder!")
        return

    results = {"success": [], "fail": []}
    for mf in modelfiles:
        if setup_model(mf):
            # Use filename without ending as model name
            name = os.path.splitext(os.path.basename(mf))[0].lower()
            results["success"].append(name)
        else:
            results["fail"].append(mf)

    print("\n" + "-"*30)
    print(f"Ready to use models: {results['success']}")
    if results["fail"]:
        print(f"Failed files: {results['fail']}")


if __name__ == "__main__":
    main()
