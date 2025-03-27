$(document).ready(function () {
    $("#upload-form").submit(function (event) {
        event.preventDefault(); // Prevent default form submission

        $("#loading").removeClass("hidden"); // Show loading animation
        $("#results").addClass("hidden"); // Hide results initially

        let formData = new FormData(this);

        $.ajax({
            url: "/predict",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                $("#loading").addClass("hidden");
                $("#results").removeClass("hidden");

                $("#prediction").text(response.prediction);
                $("#confidence").text(response.confidence);
                $("#preventive-measures").text(response.preventive_measures);
            },
            error: function () {
                alert("Error processing the image. Please try again.");
                $("#loading").addClass("hidden");
            }
        });
    });

    $(".retry-btn").click(function () {
        $("#results").addClass("hidden");
        $("#upload-form")[0].reset();
    });

    $(".consult-btn").click(function () {
        window.location.href = "/consult"; // Redirect to consultation page
    });

    // Smooth scrolling for navbar links
    $("nav a").on("click", function (event) {
        if (this.hash !== "") {
            event.preventDefault();
            $("html, body").animate(
                {
                    scrollTop: $(this.hash).offset().top,
                },
                800
            );
        }
    });
});
