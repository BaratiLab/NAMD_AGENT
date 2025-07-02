import glob
import os
import re
#fn gap
import subprocess
#fn gap
import mdtraj as md
import matplotlib.pyplot as plt
import shutil
import numpy as np



def search_dir():
    # Get only directories matching the pattern
    search_path = "/home/ach/Documents/auto_cgui/auto_cgui_master/charmm-gui-*"
    dir_matches = [d for d in glob.glob(search_path) if os.path.isdir(d)]

    if not dir_matches:
        print("No charmm-gui directories found")
        return None

    # Sort directories by modification time (latest first)
    dir_matches.sort(key=os.path.getmtime, reverse=True)
    base_dir = dir_matches[0]  # Most recently modified directory

    # Source directory to copy files from
    aux_dir = "/home/ach/Documents/auto_cgui/auto_cgui_master/aux_files"

    if not os.path.exists(aux_dir):
        print(f"Auxiliary files directory does not exist: {aux_dir}")
        return base_dir

    # Copy all files from aux_dir to base_dir
    for item in os.listdir(aux_dir):
        src_path = os.path.join(aux_dir, item)
        dst_path = os.path.join(base_dir, item)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)  # Copy with metadata

    print(f"Copied auxiliary files to {base_dir}")
    return base_dir


def clean_file(base_dir):
    """
    Clean and modify CHARMM-GUI files by:
    1. Finding the most recent charmm-gui directory
    2. Processing the size file and equilibration file
    3. Cleaning source lines and if-statements from namd files
    """
    
    def get_file_content(file_path):
        """Read and return the text content from a local file."""
        try:
            with open(file_path, "r") as file:
                return file.readlines()  # Read as a list of lines for easy manipulation
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    def parse_size_file(content):
        """
        Parse the step3_size.str content and return a dictionary mapping parameter names to values.
        Expected lines are of the form: 'SET <PARAM> = <VALUE>'
        """
        params = {}
        for line in content:
            line = line.strip()
            if line.startswith("SET"):
                parts = line.split("=")
                if len(parts) == 2:
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    tokens = key_part.split()
                    if len(tokens) >= 2:
                        param_name = tokens[1]
                        params[param_name] = value_part  # Store parameter value
        return params

    def modify_equilibration_file(equil_lines, params):
        """
        Modify the equilibration file content by:
          1. Deleting lines 7, 8, and 9.
          2. Replacing the if-statement header with "if {1} {"
          3. Updating SET command lines with values from the size file.
          4. Updating cellBasisVector lines with numeric values.
          5. Replacing instances of $a with A.
          6. Replacing instances of $zcen with the extracted value.
        """
        modified_lines = []
        
        try:
            A_val = float(params.get("A", 0))
            d_val = A_val / 2
        except ValueError:
            A_val = params.get("A", "$a")
            d_val = "$d"
        
        B_val = params.get("B", "$b")
        C_val = params.get("C", "$c")
        zcen_val = params.get("ZCEN", "$zcen")  # Get zcen value, fallback to $zcen
        
        if_pattern = re.compile(r'if\s*\{\s*\$boxtype\s*==\s*"hexa"\s*\}\s*\{')

        for index, line in enumerate(equil_lines):
            if 7 <= index <= 9:
                continue  # Skip lines 7, 8, and 9

            stripped = line.strip()
            
            if if_pattern.search(line):
                line = "if {1} {"
            
            elif stripped.startswith("SET"):
                tokens = stripped.split()
                if len(tokens) >= 2:
                    param_name = tokens[1]
                    if param_name in params:
                        line = f"SET {param_name}  = {params[param_name]}"
            
            elif stripped.startswith("cellBasisVector1"):
                line = f"cellBasisVector1     {A_val}   0.0   0.0;"
            elif stripped.startswith("cellBasisVector2"):
                line = f"cellBasisVector2     0.0    {B_val}   0.0;"
            elif stripped.startswith("cellBasisVector3"):
                line = f"cellBasisVector3    0.0   0.0    {C_val};"
            
            elif stripped.startswith("cellOrigin"):
                line = f"cellOrigin          0.0   0.0 {zcen_val};"

            # Replace instances of $a with A
            line = line.replace("$a", str(A_val))

            modified_lines.append(line)
        
        return "".join(modified_lines)

    def process_namd_file(filename):
        """Process a NAMD file to remove source lines and fix if-statements."""
        # Read all lines from the file
        with open(filename, 'r') as infile:
            lines = infile.readlines()

        # Open the file for writing to modify it in-place
        with open(filename, 'w') as outfile:
            for line in lines:
                # Skip any line that begins with "source" (ignoring leading whitespace)
                if line.lstrip().startswith("source"):
                    continue

                # Replace the specific if-statement line (ignoring indentation)
                if re.match(r'^\s*if\s*\{\s*\$boxtype\s*==\s*"hexa"\s*\}\s*\{$', line):
                    # Preserve leading whitespace
                    indentation = re.match(r'^(\s*)', line).group(1)
                    outfile.write(f'{indentation}if {{1}} {{\n')
                else:
                    outfile.write(line)

    # Process the size and equilibration files
    size_file_path = f"{base_dir}/step3_pbcsetup.str"
    equil_file_path = f"{base_dir}/namd/step4_equilibration.inp"
    
    try:
        size_content = get_file_content(size_file_path)
        equil_lines = get_file_content(equil_file_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    params = parse_size_file(size_content)
    modified_equil = modify_equilibration_file(equil_lines, params)
    
    # Write the modified content back to the same file
    with open(equil_file_path, "w") as outfile:
        outfile.write(modified_equil)
    
    print(f"Modified file has been written to {equil_file_path}")

    # Process the NAMD files
    namd_files = [
        f"{base_dir}/namd/step4_equilibration.inp",
        f"{base_dir}/namd/step5_production.inp"
    ]

    for file in namd_files:
        try:
            process_namd_file(file)
            print(f"Processed NAMD file: {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    clean_file()




import subprocess
import os

def run_simulation(base_dir, namd_path="/home/ach/Downloads/Downloads1/NAMD/NAMD_3.0_Linux-x86_64-multicore-CUDA/namd3"):
    """
    Run NAMD simulation with sequential input files.
    
    Parameters:
    base_dir (str): Base directory path containing the namd subdirectory
    namd_path (str): Path to the NAMD executable (optional, uses default if not provided)
    
    Returns:
    bool: True if all simulations completed successfully, False otherwise
    """
    
    # Define the directory containing the input files
    directory = f"{base_dir}/namd"

    # List of input files in sequential order
    input_files = [
        "step4_equilibration.inp",
        "step5_production.inp"
    ]

    # Loop through each file and run the command sequentially
    for inp_file in input_files:
        inp_basename = os.path.basename(inp_file)
        log_file = f"{inp_basename}.log"
        command = f"{namd_path} +p8 {inp_file} > {log_file}"
        print(f"Running: {command}")

        # Execute the command and wait for it to finish
        process = subprocess.run(command, shell=True, cwd=directory)
        
        if process.returncode != 0:
            print(f"Error: {inp_file} execution failed!")
            return False  # Return False if a failure occurs

    print("All jobs completed.")
    return True





def rmsd_rmsf_solution_case(base_dir, output_folder='simulation_output'):
    """
    Analyze MD trajectory and save RMSD/RMSF plots as PNG files.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the trajectory files
    output_folder : str
        Name of the output folder for saving plots (default: 'simulation_output')
    """
    
    # -------------------------
    # Setup Output Directory
    # -------------------------
    output_path = os.path.join(base_dir, output_folder)
    
    # Remove existing directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # Create fresh output directory
    os.makedirs(output_path)
    print(f"Created output directory: {output_path}")
    
    # -------------------------
    # Define File Paths
    # -------------------------
    topology_file = os.path.join(base_dir, 'namd', 'step3_input.psf')
    prod_traj_file = os.path.join(base_dir, 'namd', 'step5_production.dcd')
    
    print(f"Loading trajectory from: {prod_traj_file}")
    print(f"Using topology file: {topology_file}")
    
    # -------------------------
    # Load the Trajectory
    # -------------------------
    traj_prod = md.load(prod_traj_file, top=topology_file)
    print(f"Loaded trajectory with {traj_prod.n_frames} frames and {traj_prod.n_atoms} atoms")
    
    # -------------------------
    # Preprocessing: Align Trajectory
    # -------------------------
    print("Aligning trajectory...")
    traj_prod.superpose(traj_prod[0])
    
    # -------------------------
    # RMSD and RMSF Analysis
    # -------------------------
    print("Calculating RMSD...")
    rmsd_prod = md.rmsd(traj_prod, traj_prod, 0)
    
    print("Calculating RMSF...")
    rmsf_prod = md.rmsf(traj_prod, traj_prod[0])
    
    # -------------------------
    # Plot and Save RMSD and RMSF
    # -------------------------
    print("Generating plots...")
    
    # Create the combined plot
    plt.figure(figsize=(12, 5))
    
    # RMSD subplot
    plt.subplot(1, 2, 1)
    plt.plot(rmsd_prod, 'r-', label='RMSD', linewidth=1.5)
    plt.xlabel('Frame')
    plt.ylabel('RMSD (nm)')
    plt.title('Production RMSD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # RMSF subplot
    plt.subplot(1, 2, 2)
    plt.plot(rmsf_prod, 'b-', label='RMSF', linewidth=1.5)
    plt.xlabel('Atom Index')
    plt.ylabel('RMSF (nm)')
    plt.title('Production RMSF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(output_path, 'rmsd_rmsf_combined.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {combined_plot_path}")
    plt.close()
    
    # -------------------------
    # Save Individual Plots
    # -------------------------
    
    # RMSD plot
    plt.figure(figsize=(8, 6))
    plt.plot(rmsd_prod, 'r-', label='RMSD', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('RMSD (nm)')
    plt.title('Production RMSD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    rmsd_plot_path = os.path.join(output_path, 'rmsd_plot.png')
    plt.savefig(rmsd_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved RMSD plot: {rmsd_plot_path}")
    plt.close()
    
    # RMSF plot
    plt.figure(figsize=(8, 6))
    plt.plot(rmsf_prod, 'b-', label='RMSF', linewidth=2)
    plt.xlabel('Atom Index')
    plt.ylabel('RMSF (nm)')
    plt.title('Production RMSF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    rmsf_plot_path = os.path.join(output_path, 'rmsf_plot.png')
    plt.savefig(rmsf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved RMSF plot: {rmsf_plot_path}")
    plt.close()
    
    # -------------------------
    # Save Data as Text Files
    # -------------------------
    
    # Save RMSD data
    rmsd_data_path = os.path.join(output_path, 'rmsd_data.txt')
    with open(rmsd_data_path, 'w') as f:
        f.write("# Frame\tRMSD (nm)\n")
        for i, rmsd_val in enumerate(rmsd_prod):
            f.write(f"{i}\t{rmsd_val:.6f}\n")
    print(f"Saved RMSD data: {rmsd_data_path}")
    
    # Save RMSF data
    rmsf_data_path = os.path.join(output_path, 'rmsf_data.txt')
    with open(rmsf_data_path, 'w') as f:
        f.write("# Atom_Index\tRMSF (nm)\n")
        for i, rmsf_val in enumerate(rmsf_prod):
            f.write(f"{i}\t{rmsf_val:.6f}\n")
    print(f"Saved RMSF data: {rmsf_data_path}")
    
    print("\nAnalysis complete!")
    print(f"All outputs saved to: {output_path}")


def sasa_solution_case(base_dir, selection="protein"):
    """
    Calculate Solvent Accessible Surface Area (SASA) using VMD.
    
    Args:
        base_dir (str): Base directory path containing namd subdirectory
        selection (str): VMD selection string for SASA calculation (default: "protein")
    """
    # File paths
    dcd_file = f"{base_dir}/namd/step5_production.dcd"
    psf_file = f"{base_dir}/namd/step3_input.psf"
    output_dir = base_dir
    
    # Create the sasa.tcl script
    sasa_tcl_path = os.path.join(base_dir, "sasa.tcl")
    with open(sasa_tcl_path, "w") as f:
        f.write("""\
    ###############################################################
    # sasa.tcl                                                    #
    # DESCRIPTION:                                                #
    #    This script is quick and easy to provide procedure       #
    # for computing the Solvent Accessible Surface Area (SASA)    #
    # of Protein and allows Users to select regions of protein.   #
    #                                                             #   
    # EXAMPLE USAGE:                                              #
    #         source sasa.tcl                                     #
    #         Selection: chain A and resid 1                      #
    #                                                             #
    #   AUTHORS:                                                  #
    #	Sajad Falsafi (sajad.falsafi@yahoo.com)               #
    #       Zahra Karimi                                          # 
    #       3 Sep 2011                                             #
    ###############################################################

    # Use pre-set selmode variable
    set sel [atomselect top "$selmode"]
    set protein [atomselect top "protein"]
    set n [molinfo top get numframes]
    set output [open "$output_dir/SASA_$selmode.dat" w]
    # sasa calculation loop
    for {set i 0} {$i < $n} {incr i} {
        molinfo top set frame $i
        set sasa [measure sasa 1.4 $protein -restrict $sel]
        puts "\t \t progress: $i/$n"
        puts $output "$sasa"
    }
    puts "\t \t progress: $n/$n"
    puts "Done."	
    puts "output file: SASA_$selmode.dat"
    close $output
    """)

    # Create the wrapper TCL script
    wrapper_path = os.path.join(base_dir, "run_sasa.tcl")
    with open(wrapper_path, "w") as f:
        f.write(f"""\
    mol new "{psf_file}" type psf
    mol addfile "{dcd_file}" type dcd waitfor all
    set selmode "{selection}"
    set output_dir "{output_dir}"
    source "{sasa_tcl_path}"
    exit
    """)

    # Run VMD in text mode
    subprocess.run(["vmd", "-dispdev", "text", "-e", wrapper_path])
    
    # Plot SASA results
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Path to SASA output file
    sasa_file = f"{base_dir}/SASA_{selection}.dat"
    
    # Load the SASA data
    with open(sasa_file, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    
    # Create simulation_output directory if it doesn't exist
    output_dir = os.path.join(base_dir, "simulation_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot SASA vs. Frame
    frames = np.arange(len(data))
    
    plt.figure(figsize=(10, 6))
    plt.plot(frames, data, label="SASA", linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("SASA (Å²)")
    plt.title("Solvent Accessible Surface Area (SASA) over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot as PNG
    plot_filename = f"SASA_{selection}_plot.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()





def radius_of_gyration_solution_case(base_dir):
    # Define file paths
    psf_file = os.path.join(base_dir, "namd", "step3_input.psf")
    dcd_file = os.path.join(base_dir, "namd", "step5_production.dcd")
    rog_loop_dcd = os.path.join(base_dir, "rog_loop_dcd.tcl")

    # Create the wrapper TCL script
    wrapper_path = os.path.join(base_dir, "run_rog.tcl")
    with open(wrapper_path, "w") as f:
        f.write(f"""\
mol new "{psf_file}" type psf
mol addfile "{dcd_file}" type dcd waitfor all
source "{rog_loop_dcd}"
exit
""")

    # Run VMD in text mode
    subprocess.run(["vmd", "-dispdev", "text", "-e", wrapper_path], check=True, cwd=base_dir)


    # Load the Rg data
    rg_file = os.path.join(base_dir, "rg.dat")
    data = np.loadtxt(rg_file, skiprows=1)

    time = data[:, 0]
    rg = data[:, 1]

    # Ensure output directory exists
    output_dir = os.path.join(base_dir, "simulation_output")
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save the figure
    plt.figure(figsize=(8, 5))
    plt.plot(time, rg, label="Radius of Gyration")
    plt.xlabel("Time (or Frame)")
    plt.ylabel("Rg (Å)")
    plt.title("Radius of Gyration over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "rg_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Rg analysis complete. Plot saved to: {plot_path}")




def hydrogen_bonds_solution_case(base_dir):
    psf_file = os.path.join(base_dir, "namd", "step3_input.psf")
    dcd_file = os.path.join(base_dir, "namd", "step5_production.dcd")

    # === Create TCL Script ===
    tcl_script_path = os.path.join(base_dir, "run_hbond.tcl")
    with open(tcl_script_path, "w") as f:
        f.write(f"""\
        mol new \"{psf_file}\" type psf
        mol addfile \"{dcd_file}\" type dcd waitfor all

        package require hbonds
        hbonds -sel1 [atomselect top protein] -writefile yes -plot no

        exit
        """)

    # === Run VMD in text mode ===
    subprocess.run(["vmd", "-dispdev", "text", "-e", tcl_script_path], check=True, cwd=base_dir)

    # === Read and Plot H-bond Data ===
    hbond_data_path = os.path.join(base_dir, "hbonds.dat")
    output_dir = os.path.join(base_dir, "simulation_output")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(hbond_data_path):
        try:
            data = np.loadtxt(hbond_data_path, skiprows=1)  # Skip header
            time = data[:, 0]
            hbonds = data[:, 1]

            plt.figure(figsize=(8, 5))
            plt.plot(time, hbonds, label="H-Bonds")
            plt.xlabel("Frame")
            plt.ylabel("Number of H-Bonds")
            plt.title("Hydrogen Bonds Over Time")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(output_dir, "hbond_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved to: {plot_path}")
        except Exception as e:
            print("Could not read or plot hbonds.dat:", e)
    else:
        print("hbonds.dat not found. Please check if VMD ran correctly.")




