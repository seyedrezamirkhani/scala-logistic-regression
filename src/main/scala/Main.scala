import scala.util.Random

// Logistic Regression Implementation
object LogisticRegression {
  def run(args: Array[String]): Unit = {
    println("===== Logistic Regression =====")
    val learningRate = 0.01
    val numIterations = 1000

    // Sample dataset: (feature1, feature2, label)
    // Labels are binary: 0 or 1
    val data = Seq(
      (0.5, 0.7, 0),
      (1.5, 2.0, 0),
      (3.0, 3.5, 1),
      (2.5, 2.7, 1),
      (3.5, 4.0, 1)
    )

    // Initialize weights and bias with small random values
    var weights = Array.fill(2)(Random.nextDouble() * 0.01)
    var bias = Random.nextDouble() * 0.01

    // Sigmoid function to map predictions to probabilities
    def sigmoid(z: Double): Double = 1.0 / (1.0 + Math.exp(-z))

    // Prediction function: returns probability of label 1
    def predict(features: (Double, Double)): Double = {
      val linearModel = weights(0) * features._1 + weights(1) * features._2 + bias
      sigmoid(linearModel)
    }

    // Training loop using Gradient Descent
    for (iter <- 1 to numIterations) {
      var dw = Array(0.0, 0.0)
      var db = 0.0

      // Compute gradients
      for ((x1, x2, y) <- data) {
        val prediction = predict((x1, x2))
        val error = prediction - y

        dw(0) += error * x1
        dw(1) += error * x2
        db += error
      }

      // Update weights and bias
      for (i <- weights.indices) {
        weights(i) -= learningRate * dw(i) / data.length
      }
      bias -= learningRate * db / data.length

      // (Optional) Print loss every 100 iterations
      /*
      if (iter % 100 == 0) {
        val loss = data.map { case (x1, x2, y) =>
          val pred = predict((x1, x2))
          -y * Math.log(pred + 1e-15) - (1 - y) * Math.log(1 - pred + 1e-15)
        }.sum / data.length
        println(f"Iteration $iter: Loss = $loss%.4f")
      }
      */
    }

    // Final weights and bias
    println(s"Final weights: ${weights.mkString(", ")}")
    println(f"Final bias: $bias%.4f")

    // Calculate accuracy on the training data
    val predictions = data.map { case (x1, x2, label) =>
      val prob = predict((x1, x2))
      val predictedLabel = if (prob >= 0.5) 1 else 0
      (predictedLabel, label)
    }

    val accuracy = predictions.count { case (predicted, actual) => predicted == actual }.toDouble / data.length
    println(f"Overall accuracy on training data: ${accuracy * 100}%.2f%%")

    // Predict on new data
    val testData = Seq((1.0, 1.5), (3.0, 2.5))
    testData.foreach { case (x1, x2) =>
      val prob = predict((x1, x2))
      val predictedLabel = if (prob >= 0.5) 1 else 0
      println(s"Prediction for ($x1, $x2): $predictedLabel (probability: ${prob}%.4f)")
    }
    println()
  }
}

// Linear Regression Implementation
object LinearRegression {
  def run(args: Array[String]): Unit = {
    println("===== Linear Regression =====")
    val learningRate = 0.01
    val numIterations = 1000

    // Sample dataset: (feature1, feature2, label)
    // Labels are continuous values
    val data = Seq(
      (1.0, 2.0, 5.0),
      (2.0, 3.0, 7.0),
      (3.0, 4.0, 9.0),
      (4.0, 5.0, 11.0),
      (5.0, 6.0, 13.0)
    )

    // Initialize weights and bias with small random values
    var weights = Array.fill(2)(Random.nextDouble() * 0.01)
    var bias = Random.nextDouble() * 0.01

    // Prediction function: returns continuous output
    def predict(features: (Double, Double)): Double = {
      weights(0) * features._1 + weights(1) * features._2 + bias
    }

    // Training loop using Gradient Descent
    for (iter <- 1 to numIterations) {
      var dw = Array(0.0, 0.0)
      var db = 0.0
      var mse = 0.0

      // Compute gradients
      for ((x1, x2, y) <- data) {
        val prediction = predict((x1, x2))
        val error = prediction - y

        // Gradient of MSE with respect to weights and bias
        dw(0) += 2 * error * x1
        dw(1) += 2 * error * x2
        db += 2 * error

        // Accumulate MSE for monitoring
        mse += error * error
      }

      // Update weights and bias
      for (i <- weights.indices) {
        weights(i) -= learningRate * dw(i) / data.length
      }
      bias -= learningRate * db / data.length

      // (Optional) Print MSE every 100 iterations
      /*
      if (iter % 100 == 0) {
        val currentMSE = mse / data.length
        println(f"Iteration $iter: MSE = $currentMSE%.4f")
      }
      */
    }

    // Final weights and bias
    println(s"Final weights: ${weights.mkString(", ")}")
    println(f"Final bias: $bias%.4f")

    // Calculate Mean Squared Error on the training data
    val predictions = data.map { case (x1, x2, y) =>
      val pred = predict((x1, x2))
      (pred, y)
    }

    val mse = predictions.map { case (pred, actual) => Math.pow(pred - actual, 2) }.sum / data.length
    println(f"Mean Squared Error on training data: $mse%.4f")

    // Predict on new data
    val testData = Seq((1.0, 1.5), (3.0, 2.5))
    testData.foreach { case (x1, x2) =>
      val pred = predict((x1, x2))
      println(f"Prediction for ($x1, $x2): $pred%.4f")
    }
    println()
  }
}

// Standalone main function to run both models
object MainApp {
  def main(args: Array[String]): Unit = {
    LogisticRegression.run(args)
    LinearRegression.run(args)
  }
}

