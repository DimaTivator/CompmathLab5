import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compmath.approx import *
from compmath.interpolation import *


functions = {
    'sin': np.sin,
    'ln': np.log,
    'tan': np.tan,
    'cos': np.cos
}


def interpolate(x, y):
    # Extend x range for better visualization
    x_range = np.linspace(min(x) - 5, max(x) + 5, 200)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(x, y, label='Data Points', color='black')

    interpolation = Interpolation(list(x), list(y))

    sorted_x = np.sort(x)
    # h = np.mean(sorted_x[1:] - sorted_x[:-1])  # average step for Gauss form
    h = (max(x) - min(x)) / len(x)

    ax.plot(x_range, [interpolation.newton(val) for val in x_range], color='green', label='Newton')
    ax.plot(x_range, [interpolation.gauss(val, h) for val in x_range], color='orange', label='Gauss')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ylim(bottom=min(y) - 3, top=max(y) + 3)
    plt.legend()
    st.pyplot(fig)

    diff = interpolation.diff_y
    for i in range(len(diff)):
        for j in range(len(diff)):
            if i + j >= len(diff):
                diff[i][j] = None

    df = pd.DataFrame(diff)
    st.write(df)


# Streamlit app
def main():
    input_type = st.selectbox('Select input type', ['manual', 'file', 'function'], index=0)

    num_points = st.slider('Select the number of points (up to 15)', min_value=1, max_value=15, value=4)

    points_df = pd.DataFrame(columns=['X', 'Y'])

    x_values = []
    y_values = []

    if input_type == 'file':
        uploaded_file = st.file_uploader('Upload a file containing points (CSV format)', type='csv')
        if uploaded_file is not None:
            try:
                file_data = pd.read_csv(uploaded_file)
                x_values.extend(list(map(float, file_data['X'])))
                y_values.extend(list(map(float, file_data['Y'])))
                num_points = len(file_data)

            except Exception as e:
                st.error(f'Invalid file format. {e}')

    if input_type == 'function':
        # Get points from function
        function_name = st.selectbox('Select a function', functions.keys())
        left = st.number_input('Left', step=1.0, value=0.0)
        right = st.number_input('Right', step=1.0, value=1.0)

        if left >= right:
            st.warning('Please ensure that the value of "Left" is less than the value of "Right".')

        x_values.extend([left + i * (right - left) / (num_points - 1) for i in range(num_points)])
        y_values.extend([functions[function_name](x) for x in x_values])

    if input_type == 'manual':
        # Input fields for entering points
        st.subheader('Enter points manually:')
        for i in range(num_points):
            col1, col2 = st.columns(2)
            with col1:
                x_input = st.number_input(f'X{i + 1}', step=1.0, value=x_values[i] if i < len(x_values) else None)
                x_values.append(x_input)
            with col2:
                y_input = st.number_input(f'Y{i + 1}', step=1.0, value=y_values[i] if i < len(y_values) else None)
                y_values.append(y_input)

    # Display entered points in a table
    points_df['X'], points_df['Y'] = x_values, y_values
    st.write('Entered Points:')
    st.write(points_df)

    if st.button('Go'):
        if not points_df.empty:
            try:
                interpolate(points_df['X'].values, points_df['Y'].values)
            except Exception as e:
                st.error("Invalid data points. " + str(e))
        else:
            st.error('Please enter valid data points.')

    if not points_df.empty:
        try:
            interpolate(points_df['X'].values, points_df['Y'].values)
        except Exception as e:
            pass

    try:
        interpolation = Interpolation(list(points_df.X), list(points_df.Y))
        sorted_x = np.sort(list(points_df.X))
        # h = np.mean(sorted_x[1:] - sorted_x[:-1])  # average step for Gauss form
        h = (sorted_x[-1] - sorted_x[0]) / len(points_df)
    except Exception as e:
        pass

    method_name = st.selectbox('Select a method', ['Newton', 'Gauss'], index=0)
    point_x = st.number_input('X', step=1.0, value=0.0)
    if method_name == 'Newton':
        st.write(f'Y = {interpolation.newton(point_x):.2f}')
    else:
        st.write(f'Y = {interpolation.gauss(point_x, h):.2f}')


if __name__ == '__main__':
    main()
