from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def form_handler():
    dropdown_value = None
    numeric_value = None
    error = None

    if request.method == "POST":
        if "reset" in request.form:  # Handle Reset button
            return redirect(url_for("form_handler"))  # Clear inputs and output by redirecting

        # Handle Submit button
        dropdown_value = request.form.get("dropdown")
        numeric_value = request.form.get("numeric_input")

        print(request.form.to_dict())
        # Validate required fields
        if dropdown_value == "" or not numeric_value:
            error = "All fields are required!"

    return render_template(
        "index4.html",
        dropdown_value=dropdown_value,
        numeric_value=numeric_value,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
