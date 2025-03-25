import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# User input
v = st.slider("Velocity (v/c)", -0.99, 0.99, 0.6)
u = st.slider("Event (u/c)", -0.99, 0.99, 0.8)
t = st.number_input("Time (t)", value=5.0)

# Do Lorentz transformation
gamma = 1 / np.sqrt(1 - v**2)
x_prime = lambda x, t: gamma * (x - v * t)
ct_prime = lambda x, t: gamma * (ct - v * x)

# Plot
fig, ax = plt.subplots()
ax.set_title("Lorentz Transformation")
ax.plot([0, x_prime(1, time)], [0, ct_prime(1, time)], label="Transformed Event")
ax.legend()
st.pyplot(fig)
