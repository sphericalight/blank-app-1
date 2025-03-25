import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# User input
v_raw = st.slider("Velocity (v/c)", 0.0, 0.99, 0.6, step=0.001)   
negate1 = st.checkbox("Negative")
v = -v_raw if negate1 else v_raw

u_raw = st.slider("Event (u/c)", 0.00, 0.99, 0.80, step=0.01)
negate2 = st.checkbox("Negative")
u = -u_raw if negate2 else u_raw

ct = st.number_input("Time (ct)", value=5.0)

# A Real-Space Formulation of Special Relativity
# by Paulus Helly
# email: paulus.helly@sphericalphoton.com
# version 1.0, March 22, 2025
# ct is light-time such as light-days, light-years
# v is velocity of moving observer as the fraction to speed of light c (relative to stationary frame)
# u is velocity that define the event or velocity of second as the fraction to speed of light c (relative to stationary frame)
# show_prime is toggle; "True" for showing the primed parameter x' and ct' or "False" for suppressing them
# show_negative is toggle; "True" for showing negative x and x' or "False" for suppressing them
# set show_negative to "True" if the negative velocities are involved
x = ct * u
p_os=ct/50
show_prime=True
show_negative=True

gamma = 1
if show_prime:
    gamma = gamma / np.sqrt(1 - np.power(v, 2))

# Define figure and axis with larger size
fig = plt.figure(figsize=(10, 12))  # Increased height for better centering

# Main plot (Centered)
ax1 = fig.add_subplot(111, position=[0.15, 0.15, 0.7, 0.75])  # Adjusted to center

start_angle=0
if show_negative: #show negative x too
    ax1.set_xlim(0, np.ceil(ct*1.15))
    ax1.set_ylim(-np.ceil(ct*gamma), np.ceil(ct*1.15*gamma))
    start_angle = -np.pi/2
else:
    ax1.set_xlim(0, np.ceil(ct*1.15))
    ax1.set_ylim(0, np.ceil(ct*1.15*gamma))
ax1.set_aspect('equal')

# Draw main coordinate axes
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)
if show_negative:
    ax1.text(np.ceil(ct*1.15) -5*p_os, -3*p_os, "cτ", fontsize=10)
else:
    ax1.set_xlabel("cτ", loc='right', fontsize=10)
ax1.set_ylabel("x", loc='top', rotation=0, fontsize=10)
ax1.set_title("Real-Space", fontsize=10)

# Draw the circle (radius = ct)
theta = np.linspace(start_angle, np.pi / 2, 100)
x_circle = ct * np.cos(theta)
y_circle = ct * np.sin(theta)
ax1.plot(x_circle, y_circle, 'b', linewidth=1.2)

# Calculate angles
alpha = np.arcsin(v)
phi = np.arcsin(u)

# Draw lines for angles α and φ
x_alpha, y_alpha = ct * np.cos(alpha), ct * np.sin(alpha)
x_phi, y_phi = ct * np.cos(phi), ct * np.sin(phi)

ax1.plot([0, x_alpha], [0, y_alpha], 'g-', linewidth=1.2)
ax1.plot([0, x_phi], [0, y_phi], 'k-', linewidth=1.2)

# **Draw arcs to represent angles α and φ**
arc_radius_alpha = 0.6   # Smaller arc for α
arc_radius_phi = 1.0     # Larger arc for φ
arc_radius_phi_prime=0.7

theta_alpha = np.linspace(0, alpha, 30)  # Small arc for α
x_arc_alpha = arc_radius_alpha * np.cos(theta_alpha)
y_arc_alpha = arc_radius_alpha * np.sin(theta_alpha)
ax1.plot(x_arc_alpha, y_arc_alpha, 'g', linewidth=0.8)  # Green arc

theta_phi = np.linspace(0, phi, 30)  # Larger arc for φ
x_arc_phi = arc_radius_phi * np.cos(theta_phi)
y_arc_phi = arc_radius_phi * np.sin(theta_phi)
ax1.plot(x_arc_phi, y_arc_phi, 'k', linewidth=0.8)  # Black arc

ax1.annotate(r"$\alpha$", xy=((arc_radius_alpha+0.03) * np.cos(alpha / 2), (arc_radius_alpha+0.03) * np.sin(alpha / 2)), fontsize=9, color='green')
ax1.annotate(r"$\phi$", xy=((arc_radius_phi+0.06) * np.cos(phi / 2), (arc_radius_phi+0.06) * np.sin(phi / 2)), fontsize=9, color='black')

# Projection lines
ax1.plot([x_alpha, x_alpha], [0, y_alpha], 'g--', linewidth=0.8)
ax1.plot([ct+0.2, ct+0.2], [0, y_phi], 'k', linewidth=0.8)
ax1.plot([x_alpha, 0], [y_alpha, y_alpha], 'g--', linewidth=0.8)
ax1.plot([x_phi, 0], [y_phi, y_phi], 'k--', linewidth=0.8)
ax1.plot([x_phi+0.15, ct+0.3], [y_phi, y_phi], 'k', linewidth=0.8) #dimension line

# Projection labels
ax1.annotate("vt", xy=(x_alpha + p_os, y_alpha / 2), fontsize=9, color='green')
ax1.annotate("x", xy=(ct+0.2 + p_os, y_phi / 2), fontsize=9, color='black')
ax1.annotate(r"$c\tau_{A}$="+np.str_(f"{ct*np.cos(alpha):.5f}"), xy=(x_alpha / 2.5, y_alpha + 0.5*p_os), fontsize=9, color='green')  # cτₐ
ax1.annotate(r"$c\tau_{B}$="+np.str_(f"{ct*np.cos(phi):.5f}"), xy=(x_phi / 2.5, y_phi + 0.5*p_os), fontsize=9, color='black')  # cτᵦ

#Display the value of cTA and cTB
#ax1.annotate(r"$c\tau_{A}$="+"g", xy=(ct, 1), fontsize=9, color='black')  # cτₐ
#ax1.annotate(r"$c\tau_{B}$=", xy=(ct, 2), fontsize=9, color='black')  # cτᵦ

# Points labels
y_A= ct * np.sin(alpha)
y_B= ct * np.sin(phi)
if v<0:
    y_A=y_A-3*p_os #adjust y position of label - v
else:
    y_A=y_A+p_os

if u < 0:
    y_B=y_B-3*p_os  # adjust y position of label - u
else:
    y_B=y_B+p_os

ax1.text(0 +p_os, y_A, 'A', fontsize=11, fontweight='normal')
ax1.text(0 +p_os, y_B, 'B', fontsize=11, fontweight='normal')
ax1.text(ct * np.cos(alpha) + 0.5*p_os, y_A, "A'", fontsize=11, fontweight='normal')
ax1.text(ct * np.cos(phi) + 0.5*p_os, y_B, "B'", fontsize=11, fontweight='normal')

# ct labels
ax1.text(ct * np.cos(alpha) / 2 + v/np.abs(v)*p_os, ct * np.sin(alpha) / 2 - v/np.abs(v)*p_os, 'ct', fontsize=9)
ax1.text(ct * np.cos(phi) / 2 + u/np.abs(u)*p_os, ct * np.sin(phi) / 2 - u/np.abs(u)*p_os, 'ct', fontsize=9)

# Data for the first table (A & B)
columns_1 = ["A in O", "B in O"]
rows_1 = ["x", "v/c", "ct"]
data_1 = [[f"{ct*np.sin(alpha):.5f}", f"{ct*np.sin(phi):.5f}"], [f"{v:.5f}", f"{u:.5f}"], [f"{ct:.5f}",f"{ct:.5f}"]]

# Insert first table inside the main plot (Top-right corner)
table1 = ax1.table(cellText=data_1,
                   rowLabels=rows_1,
                   colLabels=columns_1,
                   cellLoc='right',
                   bbox=[0.50, 0.88, 0.46, 0.1])  # Lower Y value

# Adjust first table layout
table1.auto_set_font_size(False)
table1.set_fontsize(8)  # Bigger font for better readability

if show_prime:
    # Draw the ellipse
    theta = np.linspace(start_angle, np.pi / 2, 100)
    x_circle = ct * np.cos(theta)
    y_circle = ct * np.sin(theta)*gamma
    ax1.plot(x_circle, y_circle, 'r--', linewidth=1.2)

    # Draw the circle grid from focus point
    for circle_count in range (100):
        theta = np.linspace(start_angle, np.pi / 2, 100)
        x_circle = circle_count * np.cos(theta)
        y_circle = ct*np.sin(alpha)*gamma+circle_count*np.sin(theta)
        ax1.plot(x_circle, y_circle, 'k--', linewidth=0.25)

    # Data for the second table (Only A, aligned with B of table 1)
    columns_2 = ["B in O'"]
    rows_2 = ["x'", "w/c", "ct'"]
    data_2 = [[f"{(x-v*ct)*gamma:.5f}"], [f"{(u-v)/(1-u*v):.5f}"], [f"{(ct-v*x)*gamma:.5f}"]]  # Values from column B of table 1

    #add cT' axis and label
    ax1.plot([0, np.ceil(ct*1.15)], [ct*np.sin(alpha)*gamma, ct*np.sin(alpha)*gamma], 'k--', linewidth=1.2)
    ax1.text(np.ceil(ct*1.15) - 5 * p_os, ct*np.sin(alpha)*gamma-3 * p_os, "cτ'", fontsize=10)

    #annotate O'
    ax1.text(0 + p_os, ct * np.sin(alpha) * gamma - 3 * p_os, "O'", fontsize=9)

    #add ct prime
    ax1.plot([0, ct * np.cos(phi)], [ct * np.sin(alpha) * gamma, ct * np.sin(phi) * gamma], 'r', linewidth=1.5)
    ax1.annotate("ct'", xy=((ct * np.cos(phi))/2-(v*u)/np.abs(v*u)*1.2*p_os, ct*gamma*(np.sin(phi)+np.sin(alpha)) / 2+(v*u)/np.abs(v*u)*1.2*p_os), fontsize=9, color='red')

    #annotate phi_prime
    phi_prime=np.arcsin((np.sin(phi)-np.sin(alpha))/(1-np.sin(phi)*np.sin(alpha)))
    theta_phi_prime = np.linspace(0, phi_prime, 30)  # Larger arc for φ
    x_arc_phi = arc_radius_phi_prime * np.cos(theta_phi_prime)
    y_arc_phi = ct*np.sin(alpha)*gamma+arc_radius_phi_prime * np.sin(theta_phi_prime)
    ax1.plot(x_arc_phi, y_arc_phi, 'k', linewidth=0.8)  # Black arc
    ax1.annotate(r"$\phi$'", xy=((arc_radius_phi_prime+0.05)* np.cos(phi_prime / 2), ct*np.sin(alpha)*gamma+(arc_radius_phi_prime+0.05) * np.sin(phi_prime / 2)), fontsize=9, color='black')

    #add x prime
    ax1.plot([ct*np.cos(phi), ct*np.cos(phi)], [ct*np.sin(alpha)*gamma, ct*np.sin(phi)*gamma], 'r--', linewidth=0.8)
    ax1.plot([0, ct * np.cos(phi)], [ct * np.sin(phi)*gamma, ct * np.sin(phi) * gamma], 'r--', linewidth=0.8)
    ax1.annotate("x'", xy=( ct * np.cos(phi) + p_os, ct*gamma*(np.sin(phi)+np.sin(alpha)) / 2), fontsize=9, color='red')

    # ax1.set_ylim(-np.ceil(ct*gamma), np.ceil(ct*1.15*gamma))
    # add x prime scale
    #positive scale
    for x_prime_scale in range(0, np.floor(np.int_((np.ceil(ct*1.15*gamma)-ct*np.sin(alpha)*gamma)))+1):
        ax1.plot([0, 0.12], [ct*np.sin(alpha)*gamma+x_prime_scale, ct*np.sin(alpha)*gamma+x_prime_scale], 'b', linewidth=1)
        y_scale_location=ct*np.sin(alpha)*gamma+x_prime_scale-p_os
        if np.abs(y_scale_location-y_A)>0.2 and np.abs(y_scale_location-y_B)>0.2 and x_prime_scale!=0:
            ax1.text(0.18, y_scale_location, x_prime_scale, fontsize=10, color="blue")
        for minor_scale in range (4):
            ax1.plot([0, 0.08],[ct * np.sin(alpha) * gamma + x_prime_scale+minor_scale*0.25, ct * np.sin(alpha) * gamma + x_prime_scale+minor_scale*0.25], 'b', linewidth=0.8)
    #negative scale
    for x_prime_scale in range(1, np.floor(np.int_((np.ceil(ct*gamma)+ct*np.sin(alpha)*gamma)))+1):
        ax1.plot([0, 0.12], [ct*np.sin(alpha)*gamma-x_prime_scale, ct*np.sin(alpha)*gamma-x_prime_scale], 'b', linewidth=1)
        y_scale_location = ct * np.sin(alpha)*gamma-x_prime_scale - p_os
        if np.abs(y_scale_location-y_A)>0.2 and np.abs(y_scale_location-y_B)>0.2 and x_prime_scale!=0:
            ax1.text(0.18, y_scale_location, -x_prime_scale, fontsize=10, color="blue")
        for minor_scale in range (4):
            ax1.plot([0, 0.08],[ct * np.sin(alpha) * gamma - x_prime_scale+minor_scale*0.25, ct * np.sin(alpha) * gamma - x_prime_scale+minor_scale*0.25], 'b', linewidth=0.8)

    table2 = ax1.table(cellText=data_2,
                       rowLabels=rows_2,
                       colLabels=columns_2,
                       cellLoc='right',
                       bbox=[0.73, 0.76, 0.23, 0.1])  # Lower Y value

    # Adjust second table layout to match first
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)

    # Adjust column headers for table2
    for key, cell in table2.get_celld().items():
        if key[0] == 0:  # Row index 0 is for column headers
            cell.set_text_props(ha='right', va='center', fontweight='normal')  # right align

# Adjust column headers for table1
for key, cell in table1.get_celld().items():
    if key[0] == 0:  # Row index 0 is for column headers
        cell.set_text_props(ha='right', va='center', fontweight='normal')  # right align


# Grid for main plot
ax1.grid(True, linestyle='--', alpha=0.5)

# Adjust layout for centering
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)  # Centers the plot

st.pyplot(fig)
