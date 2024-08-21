import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import pyperclip

print("Starting the linear regression script...")

# Initialize global data variables
data = {'Month': [], 'Sales': []}
num_months = 6  # Default value for the number of past months

def set_num_months():
    global num_months
    try:
        num_months = int(entry_num_months.get())
        print(f"Number of past months set to: {num_months}")
    except ValueError:
        print("Invalid number of past months entered. Please enter an integer.")

def paste_dates():
    global data
    clipboard = pyperclip.paste()
    dates = clipboard.split()
    data['Month'] = list(map(int, dates))
    print(f"Pasted Dates: {data['Month']}")

def paste_values():
    global data
    clipboard = pyperclip.paste()
    values = clipboard.split()
    data['Sales'] = list(map(int, values))
    print(f"Pasted Values: {data['Sales']}")
    process_data()

def process_data():
    global num_months
    if len(data['Month']) < num_months or len(data['Sales']) < num_months:
        print(f"Data is incomplete. Need at least {num_months} months of data.")
        return

    df = pd.DataFrame(data)
    print("Data loaded successfully.")

    # Define the independent and dependent variables
    X = df[['Month']]
    y = df['Sales']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Create and train the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    print("Linear model training completed.")

    # Make predictions using the linear regression model
    y_pred_linear = linear_model.predict(X_test)

    # Evaluate the linear regression model
    slope = linear_model.coef_[0]
    intercept = linear_model.intercept_
    r_squared = linear_model.score(X_test, y_test)

    # Extend the range of months for forecasting
    last_month = max(data['Month'])
    future_months = list(range(last_month + 1, last_month + 13))
    future_sales_linear = linear_model.predict(np.array(future_months).reshape(-1, 1))

    # Calculate the average sales of the past num_months
    average_sales = round(np.mean(y[-num_months:]))

    # Create the plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Actual Sales')
    ax.plot(X, linear_model.predict(X), color='red', label='Linear Regression Line')
    ax.scatter(future_months, future_sales_linear, color='green', label='Forecasted Sales (Linear)')
    ax.plot(future_months, future_sales_linear, color='orange', linestyle='dashed', label='Forecasted Trend (Linear)')
    ax.axhline(y=average_sales, color='purple', linestyle='dashed', label=f'Average Sales (Last {num_months} Months): {average_sales}')

    # Add labels for the values and months (excluding average sales line)
    for i in range(len(X)):
        ax.annotate(f'{y[i]}', (X.iloc[i, 0], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        ax.annotate(f'{X.iloc[i, 0]}', (X.iloc[i, 0], linear_model.predict(X)[i]), textcoords="offset points", xytext=(0, -15), ha='center', color='red')

    for i in range(len(future_months)):
        ax.annotate(f'{int(future_sales_linear[i])}', (future_months[i], future_sales_linear[i]), textcoords="offset points", xytext=(0, 10), ha='center', color='green')

    ax.set_xlabel('Month')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Forecasting using Linear Regression and Average Sales')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Display the results
    ttk.Label(frame_results, text=f"Linear Model Coefficient (Slope): {slope:.4f}").pack(pady=5)
    ttk.Label(frame_results, text=f"Linear Model Intercept: {intercept:.4f}").pack(pady=5)
    ttk.Label(frame_results, text=f"Linear R-squared: {r_squared:.4f}").pack(pady=5)
    ttk.Label(frame_results, text=f"Average Sales (Last {num_months} Months): {average_sales}").pack(pady=5)

    # Display the forecasted sales in a table
    columns = ['Month'] + [f'{month}' for month in range(1, last_month + 13)]
    tree = ttk.Treeview(frame_results, columns=columns, show='headings')
    tree.pack(pady=5)

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=60, anchor='center')

    past_sales = ['Past Sales'] + data['Sales'] + [''] * (num_months - len(data['Sales']))
    forecasted_sales_linear = ['Forecasted Sales (Linear)'] + [''] * len(data['Sales']) + list(map(int, future_sales_linear))
    average_sales_list = ['Average Sales'] + [''] * len(data['Sales']) + [average_sales] * 12

    tree.insert('', 'end', values=past_sales)
    tree.insert('', 'end', values=forecasted_sales_linear)
    tree.insert('', 'end', values=average_sales_list)

    # Add a button to download results to Excel
    ttk.Button(frame_buttons, text="Download to Excel", command=lambda: download_to_excel(df, future_months, future_sales_linear, average_sales)).pack(side=tk.LEFT, padx=5, pady=5)

def download_to_excel(df, future_months, future_sales_linear, average_sales):
    # Create DataFrames for forecasted sales
    future_df_linear = pd.DataFrame({'Month': future_months, 'Sales (Linear)': future_sales_linear})
    future_df_avg = pd.DataFrame({'Month': future_months, 'Sales (Average)': [average_sales] * 12})

    # Combine past and forecasted sales
    combined_df = pd.concat([df, future_df_linear['Sales (Linear)'], future_df_avg['Sales (Average)']], axis=1)
    # Save to Excel file
    file_path = 'C:\\Users\\Frank\\Desktop\\predicted_sales_forecast.xlsx'
    combined_df.to_excel(file_path, index=False)
    print(f"Data saved to {file_path}")

# Create a Tkinter GUI to display the results
root = tk.Tk()
root.title("Linear Regression Results")
root.geometry("1200x800")  # Resize the window to fit the new text

# Create a frame for the plot
frame_plot = ttk.Frame(root)
frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create a frame for the buttons
frame_buttons = ttk.Frame(root)
frame_buttons.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Add buttons for pasting dates and values
ttk.Button(frame_buttons, text="Paste Dates", command=paste_dates).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_buttons, text="Paste Values", command=paste_values).pack(side=tk.LEFT, padx=5, pady=5)

# Add an entry field and button to set the number of past months
ttk.Label(frame_buttons, text="Number of past months:").pack(side=tk.LEFT, padx=5, pady=5)
entry_num_months = ttk.Entry(frame_buttons, width=5)
entry_num_months.pack(side=tk.LEFT, padx=5, pady=5)
ttk.Button(frame_buttons, text="Set", command=set_num_months).pack(side=tk.LEFT, padx=5, pady=5)
ttk.Label(frame_buttons, text='Input the number of past months and click "Set" before pasting Dates and Values.').pack(side=tk.LEFT, padx=5, pady=5)

# Create a frame for the results
frame_results = ttk.Frame(root)
frame_results.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

# Start the Tkinter main loop
root.mainloop()
