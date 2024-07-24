unit = input("Is the Temperature in Celsius, Fahrenheit, or Kelvin? (C/F/K): ").lower()
temp = float(input("Enter the temperature: "))

if unit == "C":
    fahrenheit = round((9 * temp) / 5 + 32, 2)
    kelvin = round(temp + 273.15, 2)
    print(f"{temp} degrees Celsius is equal to {fahrenheit} degrees Fahrenheit and {kelvin} degrees Kelvin.")
elif unit == "F":
    celsius = round((temp - 32) * 5 / 9, 2)
    kelvin = round((temp - 32) * 5 / 9 + 273.15, 2)
    print(f"{temp} degrees Fahrenheit is equal to {celsius} degrees Celsius and {kelvin} degrees Kelvin.")
elif unit == "K":
    celsius = round(temp - 273.15, 2)
    fahrenheit = round((temp - 273.15) * 9 / 5 + 32, 2)
    print(f"{temp} degrees Kelvin is equal to {celsius} degrees Celsius and {fahrenheit} degrees Fahrenheit.")
else:
    print(f"{unit} is invalid for measurement.")
