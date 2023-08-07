import csv
import matplotlib.pyplot as plt

def open_csv(file, name):

    data = []

    with open(file, mode='r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] == name:
                current_data = row[2:]
                for i in range(len(current_data)):
                    if current_data[i] != '':
                        current_data[i] = float(current_data[i])
                    else:
                        current_data[i] = None

        data.append(current_data)

    return data[0]


def plot_time_series(y_values):
    # Assuming the time series data has regularly spaced time intervals, starting from 0
    x_values = list(range(len(y_values)))


    # Plot the time series data
    plt.plot(x_values, y_values)

    # Set labels for the plot
    plt.xlabel("Frames")
    plt.ylabel("Wrist Flexion Angle (Degrees)")
    plt.title("Wrist Flexion Angle over Time")

    # Show the plot
    plt.show()

def main():
    bicep_curl_data = open_csv('../bicep_outputs/wrist_flexion_angles.csv', 'bicep_good3.MOV')
    plot_time_series(bicep_curl_data)

main()