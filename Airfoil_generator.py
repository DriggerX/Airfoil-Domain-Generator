import tkinter as tk
from tkinter import messagebox
import random
import cadquery as cq
import numpy as np
import matplotlib.pyplot as plt
import os

def show_generation_options():
    generation_type_label.pack(anchor="w")
    random_radio.pack(anchor="w")
    customized_radio.pack(anchor="w")

def show_customized_input():
    customized_digits_entry.pack(anchor="w")
    num_airfoils_label.pack_forget()
    num_airfoils_entry.pack_forget()
    generate_button.pack()

def show_random_input():
    customized_digits_entry.pack_forget()
    num_airfoils_label.pack(anchor="w")
    num_airfoils_entry.pack(anchor="w")
    generate_button.pack()

def naca4_airfoil(naca_code, c=1.0, n_points=200):
    m = int(naca_code[0]) / 100.0  # maximum camber as percentage of chord
    p = int(naca_code[1]) / 10.0   # position of maximum camber from leading edge in tenths of chord
    t = int(naca_code[2:]) / 100.0 # maximum thickness as percentage of chord

    def camber_line(x):
        if p != 0:
            if x < p*c:
                return m * (x / (p**2)) * (2*p - x / c)
            else:
                return m * ((c - x) / ((1 - p)**2)) * (1 + x/c - 2*p)
        else:
            return 0.0
    
    def thickness_distribution(x):
        return 5 * t * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
    
    x = np.linspace(0, c, n_points)
    yt = thickness_distribution(x)
    yc = np.array([camber_line(xi) for xi in x])
    
    xu = x - yt * np.sin(np.arctan(np.gradient(yc, x)))
    yu = yc + yt * np.cos(np.arctan(np.gradient(yc, x)))
    xl = x + yt * np.sin(np.arctan(np.gradient(yc, x)))
    yl = yc - yt * np.cos(np.arctan(np.gradient(yc, x)))
    
    upper_surface = np.column_stack((xu, yu))
    lower_surface = np.column_stack((xl, yl))
    
    return np.vstack((upper_surface, lower_surface[::-1]))

# FOR NACA 5 SERIES

def naca5_airfoil(naca_code, c=1.0, n_points=100):
    cl = int(naca_code[0]) * 0.15  # Theoretical lift coefficient
    p = int(naca_code[1]) / 2.0    # Position of maximum camber
    reflex = int(naca_code[2])     # Reflex camber flag
    t = int(naca_code[3:]) / 100.0 # Maximum thickness

    def camber_line(x):
        m = cl / 0.15  # Derived camber line coefficient
        if reflex == 0:
            if x < p*c:
                return m * (x / (p**2)) * (2*p - x / c)
            else:
                return m * ((c - x) / ((1 - p)**2)) * (1 + x/c - 2*p)
        else:
            # Reflex camber line equations
            k1 = 15.957  
            k2 = -0.1036
            if x < p*c:
                return k1 * (x / (p**3)) * (3*p - x/c) + k2 * (x / (p**3)) * (x/c - p)
            else:
                return k1 * ((c - x) / ((1 - p)**3)) * (1 + x/c - 3*p) + k2 * ((c - x) / ((1 - p)**3)) * (1 + x/c - p)

    def thickness_distribution(x):
        return 5 * t * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
    
    x = np.linspace(0, c, n_points)
    yt = thickness_distribution(x)
    yc = np.array([camber_line(xi) for xi in x])
    
    xu = x - yt * np.sin(np.arctan(np.gradient(yc, x)))
    yu = yc + yt * np.cos(np.arctan(np.gradient(yc, x)))
    xl = x + yt * np.sin(np.arctan(np.gradient(yc, x)))
    yl = yc - yt * np.cos(np.arctan(np.gradient(yc, x)))
    
    upper_surface = np.column_stack((xu, yu))
    lower_surface = np.column_stack((xl, yl))
    
    return np.vstack((upper_surface, lower_surface[::-1]))


def naca6_airfoil(series, min_pressure_pos, lift_coeff_range, design_lift_coeff, max_thickness, laminar_flow_fraction, c=1.0, n_points=200):
    
    
    def camber_line(x):
        if x < min_pressure_pos * c:
            return design_lift_coeff * (x / (min_pressure_pos**2)) * (2 * min_pressure_pos - x / c)
        else:
            return design_lift_coeff * ((c - x) / ((1 - min_pressure_pos)**2)) * (1 + x / c - 2 * min_pressure_pos)
    
    def thickness_distribution(x):
        return 5 * max_thickness * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)
    
    x = np.linspace(0, c, n_points)
    yt = thickness_distribution(x)
    yc = np.array([camber_line(xi) for xi in x])
    
    xu = x - yt * np.sin(np.arctan(np.gradient(yc, x)))
    yu = yc + yt * np.cos(np.arctan(np.gradient(yc, x)))
    xl = x + yt * np.sin(np.arctan(np.gradient(yc, x)))
    yl = yc - yt * np.cos(np.arctan(np.gradient(yc, x)))
    
    upper_surface = np.column_stack((xu, yu))
    lower_surface = np.column_stack((xl, yl))
    
    return np.vstack((upper_surface, lower_surface[::-1]))


def generate_airfoil():
    series = airfoil_series_var.get()
    generation_type = generation_type_var.get()

    if series == "":
        messagebox.showerror("Error", "Please select an airfoil series.")
        return
    
    if generation_type == "":
        messagebox.showerror("Error", "Please select generation type (Random or Customized).")
        return

    digits = ""
    output_files = []
    output_dir = "Airfoils_databases_generated"
    os.makedirs(output_dir, exist_ok=True)

    if generation_type == "Random":
        try:
            num_airfoils = int(num_airfoils_entry.get())
            if num_airfoils <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of airfoils.")
            return

        if series == "4 Series":
            for i in range(num_airfoils):
                first_digit = np.random.randint(0, 10)
                second_digit = np.random.randint(0, 10)
                last_two_digits = np.random.randint(10, 25)
                naca_code = f"{first_digit}{second_digit}{last_two_digits:02d}"
                airfoil_points = naca4_airfoil(naca_code)
                airfoil = cq.Workplane("XY").polyline(airfoil_points).close().extrude(1)
                file_path = os.path.join(output_dir, f"naca_{naca_code}.step")
                airfoil.val().exportStep(file_path)
                output_files.append(file_path)
            result_label.config(text=f"Generated {num_airfoils} random NACA 4-Series airfoils.")
        elif series == "5 Series":
            for i in range(num_airfoils):
                first_digit = np.random.randint(2, 4)
                second_digit = np.random.randint(0, 10)
                third_digit = np.random.randint(0, 2)
                last_two_digits = np.random.randint(10, 25)
                naca_code = f"{first_digit}{second_digit}{third_digit}{last_two_digits:02d}"
                airfoil_points = naca5_airfoil(naca_code)
                airfoil = cq.Workplane("XY").polyline(airfoil_points).close().extrude(1)
                file_path = os.path.join(output_dir, f"naca_{naca_code}.step")
                airfoil.val().exportStep(file_path)
                output_files.append(file_path)
            result_label.config(text=f"Generated {num_airfoils} random NACA 5-Series airfoils.")
        elif series == "6 Series":
            for i in range(num_airfoils):
                min_pressure_pos = round(np.random.uniform(0.3, 0.8), 1)
                lift_coeff_range = round(np.random.uniform(0.5, 1.5), 1)
                design_lift_coeff = round(np.random.uniform(0.5, 1.0), 2)
                max_thickness = round(np.random.uniform(0.1, 0.25), 2)
                laminar_flow_fraction = round(np.random.uniform(0.3, 1.0), 1)
                airfoil_points = naca6_airfoil(6, min_pressure_pos, lift_coeff_range, design_lift_coeff, max_thickness, laminar_flow_fraction)
                airfoil = cq.Workplane("XY").polyline(airfoil_points).close().extrude(1)
                file_path = os.path.join(output_dir, f"naca_6_{min_pressure_pos}_{lift_coeff_range}_{design_lift_coeff}_{max_thickness}_{laminar_flow_fraction}.step")
                airfoil.val().exportStep(file_path)
                output_files.append(file_path)
            result_label.config(text=f"Generated {num_airfoils} random NACA 6-Series airfoils.")
    elif generation_type == "Customized":
        digits = customized_digits_entry.get().strip()
        if not digits.isdigit():
            messagebox.showerror("Error", "Please enter a valid NACA code.")
            return
        if series == "4 Series":
            if len(digits) != 4:
                messagebox.showerror("Error", "Please enter a valid 4-digit NACA code.")
                return
            airfoil_points = naca4_airfoil(digits)
            airfoil = cq.Workplane("XY").polyline(airfoil_points).close().extrude(1)
            file_path = os.path.join(output_dir, f"naca_{digits}.step")
            airfoil.val().exportStep(file_path)
            output_files.append(file_path)
            result_label.config(text=f"Generated NACA 4-Series airfoil with code {digits}.")
        elif series == "5 Series":
            if len(digits) != 5:
                messagebox.showerror("Error", "Please enter a valid 5-digit NACA code.")
                return
            airfoil_points = naca5_airfoil(digits)
            airfoil = cq.Workplane("XY").polyline(airfoil_points).close().extrude(1)
            file_path = os.path.join(output_dir, f"naca_{digits}.step")
            airfoil.val().exportStep(file_path)
            output_files.append(file_path)
            result_label.config(text=f"Generated NACA 5-Series airfoil with code {digits}.")
        elif series == "6 Series":
            try:
                min_pressure_pos, lift_coeff_range, design_lift_coeff, max_thickness, laminar_flow_fraction = map(float, digits.split())
                airfoil_points = naca6_airfoil(6, min_pressure_pos, lift_coeff_range, design_lift_coeff, max_thickness, laminar_flow_fraction)
                airfoil = cq.Workplane("XY").polyline(airfoil_points).close().extrude(1)
                file_path = os.path.join(output_dir, f"naca_6_{min_pressure_pos}_{lift_coeff_range}_{design_lift_coeff}_{max_thickness}_{laminar_flow_fraction}.step")
                airfoil.val().exportStep(file_path)
                output_files.append(file_path)
                result_label.config(text=f"Generated NACA 6-Series airfoil with parameters {digits}.")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid parameters for the NACA 6-Series airfoil.")
                return
    messagebox.showinfo("Generation Complete", f"Airfoil(s) generated and saved to: {', '.join(output_files)}")


root = tk.Tk()
root.title("NACA Airfoil Generator")

title_label = tk.Label(root, text="NACA Airfoil Generator", font=("Helvetica", 16))
title_label.pack()

airfoil_series_label = tk.Label(root, text="Select Airfoil Series:")
airfoil_series_label.pack(anchor="w")

airfoil_series_var = tk.StringVar(value="")
series_4_radio = tk.Radiobutton(root, text="NACA 4 Series", variable=airfoil_series_var, value="4 Series")
series_4_radio.pack(anchor="w")
series_5_radio = tk.Radiobutton(root, text="NACA 5 Series", variable=airfoil_series_var, value="5 Series")
series_5_radio.pack(anchor="w")
series_6_radio = tk.Radiobutton(root, text="NACA 6 Series", variable=airfoil_series_var, value="6 Series")
series_6_radio.pack(anchor="w")

generation_type_label = tk.Label(root, text="Select Generation Type:")
generation_type_var = tk.StringVar(value="")
random_radio = tk.Radiobutton(root, text="Random", variable=generation_type_var, value="Random", command=show_random_input)
customized_radio = tk.Radiobutton(root, text="Customized", variable=generation_type_var, value="Customized", command=show_customized_input)

customized_digits_entry = tk.Entry(root, width=50)
num_airfoils_label = tk.Label(root, text="Number of Airfoils to Generate:")
num_airfoils_entry = tk.Entry(root, width=10)

generate_button = tk.Button(root, text="Generate Airfoil", command=generate_airfoil)

result_label = tk.Label(root, text="", font=("Helvetica", 12))
result_label.pack()

show_generation_options()


root.mainloop()
