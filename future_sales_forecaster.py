import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import pyperclip

print("Starting the linear regression script...")

# ----------------------------
# Global Data and Variables
# ----------------------------
data = {'Month': [], 'Sales': []}  # Expects month strings like "Jan-2023"
num_months = 6  # Default number of past months for averaging

# Global variables to hold processed data for download
current_df = None
current_future_dates = None
current_future_sales = None
current_average_sales = None

# Reference to the Download button and message variable
download_button = None
message_var = None

# ----------------------------
# Function Definitions
# ----------------------------
def set_num_months():
    global num_months
    try:
        num_months = int(entry_num_months.get())
        message_var.set(f"Number of past months set to: {num_months}")
        print(f"Number of past months set to: {num_months}")
    except ValueError:
        message_var.set("Invalid number of past months entered. Please enter an integer.")
        print("Invalid number of past months entered. Please enter an integer.")

def paste_dates():
    global data
    clipboard = pyperclip.paste()
    dates = clipboard.split()
    data['Month'] = dates
    message_var.set("Dates pasted successfully.")
    print(f"Pasted Dates: {data['Month']}")

def paste_values():
    global data
    clipboard = pyperclip.paste()
    values = clipboard.split()
    try:
        data['Sales'] = list(map(int, values))
        message_var.set("Values pasted successfully.")
        print(f"Pasted Values: {data['Sales']}")
        process_data()
    except ValueError:
        message_var.set("Invalid sales values. Ensure all values are integers.")
        print("Invalid sales values. Ensure all values are integers.")

def process_data():
    global num_months, download_button
    global current_df, current_future_dates, current_future_sales, current_average_sales

    if len(data['Month']) < num_months or len(data['Sales']) < num_months:
        message_var.set(f"Data is incomplete. Need at least {num_months} months of data.")
        print(f"Data is incomplete. Need at least {num_months} months of data.")
        return

    df = pd.DataFrame(data)
    print("Data loaded successfully.")

    try:
        df['Date'] = pd.to_datetime(df['Month'], format='%b-%Y')
    except Exception as e:
        message_var.set(f"Error parsing dates: {e}")
        print(f"Error parsing dates. Ensure they are in the format 'Jan-2023': {e}")
        return

    df['Date_num'] = mdates.date2num(df['Date'])
    X = df[['Date_num']]
    y = df['Sales']

    linear_model = LinearRegression()
    linear_model.fit(X, y)
    print("Linear model training completed.")

    y_pred_linear = linear_model.predict(X)
    slope = linear_model.coef_[0]
    intercept = linear_model.intercept_
    r_squared = linear_model.score(X, y)

    last_date = df['Date'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
    future_dates_num = mdates.date2num(future_dates)
    future_sales_linear = linear_model.predict(np.array(future_dates_num).reshape(-1, 1))
    average_sales = round(np.mean(y[-num_months:]))

    # Save processed data for download usage.
    current_df = df.copy()
    current_future_dates = future_dates.copy()
    current_future_sales = future_sales_linear.copy()
    current_average_sales = average_sales

    # ----------------------------
    # Create and Display the Chart (in a Separate Window)
    # ----------------------------
    fig, ax = plt.subplots()

    ax.scatter(df['Date'], y, color='blue', label='Actual Sales')
    ax.plot(df['Date'], y_pred_linear, color='red', label='Linear Regression Line')

    ax.scatter(future_dates, future_sales_linear, color='green', zorder=5,
               label='Forecasted Sales (Linear)')
    ax.plot(future_dates, future_sales_linear, color='orange', linestyle='dashed',
            label='Forecasted Trend (Linear)')

    ax.axhline(y=average_sales, color='purple', linestyle='dashed',
               label=f'Average Sales (Last {num_months} Months): {average_sales}')

    for i in range(len(df)):
        offset = 10 if i % 2 == 0 else -15
        va = 'bottom' if i % 2 == 0 else 'top'
        ax.annotate(
            text=f'{y.iloc[i]}',
            xy=(df['Date'].iloc[i], y.iloc[i]),
            xytext=(0, offset),
            textcoords="offset points",
            ha='center',
            va=va,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
        )

    for i in range(len(future_dates)):
        if i % 3 == 0:
            offset = 10 if i % 2 == 0 else -15
            va = 'bottom' if i % 2 == 0 else 'top'
            ax.annotate(
                text=f'{int(future_sales_linear[i])}',
                xy=(future_dates[i], future_sales_linear[i]),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center',
                va=va,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
            )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    fig.autofmt_xdate()

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Sales', fontsize=10)
    ax.set_title('Sales Forecasting using Linear Regression and Average Sales', fontsize=10)
    ax.legend(fontsize=9)

    chart_window = tk.Toplevel(root)
    chart_window.title("Sales Forecast Chart")
    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ----------------------------
    # Update the Results Table (with Horizontal Scrollbar)
    # ----------------------------
    for widget in frame_results.winfo_children():
        widget.destroy()

    ttk.Label(frame_results, text=f"Linear Model Coefficient (Slope): {slope:.4f}").pack(pady=2)
    ttk.Label(frame_results, text=f"Linear Model Intercept: {intercept:.4f}").pack(pady=2)
    ttk.Label(frame_results, text=f"Linear R-squared: {r_squared:.4f}").pack(pady=2)
    ttk.Label(frame_results, text=f"Average Sales (Last {num_months} Months): {average_sales}").pack(pady=2)

    past_dates_str = df['Date'].dt.strftime('%b-%Y').tolist()
    future_dates_str = [d.strftime('%b-%Y') for d in future_dates]
    columns = ['Type'] + past_dates_str + future_dates_str

    # Create a Canvas to hold the Treeview so the horizontal scrollbar is always visible.
    tree_canvas = tk.Canvas(frame_results, height=100)
    tree_canvas.pack(fill=tk.X, expand=False)
    h_scroll = ttk.Scrollbar(frame_results, orient="horizontal", command=tree_canvas.xview)
    h_scroll.pack(fill=tk.X)
    tree_canvas.configure(xscrollcommand=h_scroll.set)

    tree_frame = ttk.Frame(tree_canvas)
    tree_canvas.create_window((0, 0), window=tree_frame, anchor="nw")

    tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=4)
    tree.pack(fill=tk.X)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=80, anchor='center')

    past_sales_row = ['Past Sales'] + data['Sales'] + [''] * len(future_dates_str)
    forecasted_sales_row = ['Forecasted Sales (Linear)'] + [''] * len(data['Sales']) + list(map(lambda x: int(round(x)), future_sales_linear))
    average_sales_row = ['Average Sales'] + [''] * len(data['Sales']) + [average_sales] * len(future_dates_str)
    tree.insert('', 'end', values=past_sales_row)
    tree.insert('', 'end', values=forecasted_sales_row)
    tree.insert('', 'end', values=average_sales_row)

    # Auto-size the first column using the default system font.
    default_font = tkFont.nametofont("TkDefaultFont")
    max_width = 0
    for item in tree.get_children():
        text = tree.set(item, columns[0])
        width = default_font.measure(text)
        if width > max_width:
            max_width = width
    tree.column(columns[0], width=max_width + 10)

    tree_frame.update_idletasks()
    tree_canvas.configure(scrollregion=tree_canvas.bbox("all"))
    tree_canvas.config(width=1000)

    # Update the download button command and enable it.
    download_button.config(command=lambda: download_to_excel(current_df, current_future_dates, current_future_sales, current_average_sales), state="normal")
    print("Download button enabled.")

def download_to_excel(df, future_dates, future_sales_linear, average_sales):
    df_forecast = pd.DataFrame({
        'Date': future_dates,
        'Sales (Linear)': future_sales_linear,
        'Sales (Average)': [average_sales] * len(future_dates)
    })
    df_export = pd.concat([df[['Date', 'Sales']], df_forecast], ignore_index=True)
    file_path = 'C:\\Users\\Frank\\Desktop\\predicted_sales_forecast.xlsx'
    df_export.to_excel(file_path, index=False)
    print(f"Data saved to {file_path}")

def open_about_window():
    about_win = tk.Toplevel(root)
    about_win.title("About Number of Past Months")
    about_text = (
        "- User sets the number of recent months to include in the average sales calculation.\n"
        "- A horizontal dashed line shows the average sales over the selected period.\n"
        "- The calculated average is listed as a baseline measure.\n"
        "- Assists in comparing the linear regression trend against recent performance.\n"
        "- Can be simplified or removed if the feature isn’t valuable for larger datasets or when using the entire sales history."
    )
    label = ttk.Label(about_win, text=about_text, justify=tk.LEFT)
    label.pack(padx=10, pady=10)

# ----------------------------
# Main GUI Setup (Single Main Window)
# ----------------------------
root = tk.Tk()
root.title("Future Sales Forecaster")
root.geometry("1000x380")  # Compact main window

# Set up custom style for buttons.
style = ttk.Style()
style.theme_use('clam')
style.configure('Custom.TButton', background='#d0e8f1', foreground='black')
style.map('Custom.TButton', background=[('active', '#87CEFA')], foreground=[('active', 'black')])

# Message variable for status updates.
message_var = tk.StringVar()
message_var.set("Begin by pasting Dates and Values from Excel.")

frame_buttons = ttk.Frame(root)
frame_buttons.pack(side=tk.TOP, fill=tk.X, expand=False)

# Message label for status notifications.
message_label = ttk.Label(frame_buttons, textvariable=message_var, foreground="green")
message_label.pack(side=tk.TOP, padx=5, pady=2)

frame_buttons_row1 = ttk.Frame(frame_buttons)
frame_buttons_row1.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
ttk.Button(frame_buttons_row1, text="Paste Dates", command=paste_dates, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
ttk.Button(frame_buttons_row1, text="Paste Values", command=paste_values, style='Custom.TButton').pack(side=tk.LEFT, padx=5)

frame_buttons_row2 = ttk.Frame(frame_buttons)
frame_buttons_row2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
ttk.Label(frame_buttons_row2, text="Number of past months:").pack(side=tk.LEFT, padx=5)
entry_num_months = ttk.Entry(frame_buttons_row2, width=5)
entry_num_months.pack(side=tk.LEFT, padx=5)
ttk.Button(frame_buttons_row2, text="Set", command=set_num_months, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
ttk.Label(frame_buttons_row2, text='Click "Set" before pasting Dates and Values.').pack(side=tk.LEFT, padx=5)

frame_buttons_row3 = ttk.Frame(frame_buttons)
frame_buttons_row3.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
ttk.Button(frame_buttons_row3, text="About Number of Past Months", command=open_about_window, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
download_button = ttk.Button(frame_buttons_row3, text="Download to Excel", state="disabled", style='Custom.TButton')
download_button.pack(side=tk.LEFT, padx=5)

frame_results = ttk.Frame(root)
frame_results.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

root.mainloop()
