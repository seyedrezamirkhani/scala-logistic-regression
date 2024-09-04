import scala.util.Random

// Logistic Regression Implementation
object LogisticRegression {

  def run(args: Array[String]): Unit = {
    println("===== Logistic Regression =====")
    val learningRate = 0.01
    val numIterations = 1000

    // Sample training dataset: (feature1, feature2, label)
    val trainingData = Seq(
      (0.5, 0.7, 0),
      (1.5, 2.0, 0),
      (3.0, 3.5, 1),
      (2.5, 2.7, 1),
      (3.5, 4.0, 1)
    )

    // Sample test dataset: (feature1, feature2, label)
    val testData = Seq(
      (1.0, 1.5, 0),
      (3.0, 2.5, 1),
      (2.0, 2.2, 0),
      (4.0, 4.5, 1)
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

    // Function to calculate accuracy
    def calculateAccuracy(data: Seq[(Double, Double, Int)]): Double = {
      val predictions = data.map { case (x1, x2, label) =>
        val prob = predict((x1, x2))
        val predictedLabel = if (prob >= 0.5) 1 else 0
        (predictedLabel, label)
      }
      predictions.count { case (predicted, actual) => predicted == actual }.toDouble / data.length
    }


    // Training loop using Gradient Descent
    for (_ <- 1 to numIterations) {
      var dw = Array(0.0, 0.0)
      var db = 0.0

      // Compute gradients
      for ((x1, x2, y) <- trainingData) {
        val prediction = predict((x1, x2))
        val error = prediction - y

        dw(0) += error * x1
        dw(1) += error * x2
        db += error
      }

      // Update weights and bias
      for (i <- weights.indices) {
        weights(i) -= learningRate * dw(i) / trainingData.length
      }
      bias -= learningRate * db / trainingData.length
    }

    // Final weights and bias
    println(s"Final weights: ${weights.mkString(", ")}")
    println(f"Final bias: $bias%.4f")

    // Calculate accuracy on the training data
    val trainingAccuracy = calculateAccuracy(trainingData)
    println(f"Overall accuracy on training data: ${trainingAccuracy * 100}%.2f%%")

    // Calculate accuracy on the test data
    val testAccuracy = calculateAccuracy(testData)
    println(f"Overall accuracy on test data: ${testAccuracy * 100}%.2f%%")

    // Predict on new data
    val testDataForPrediction = Seq((1.0, 1.5), (3.0, 2.5))
    testDataForPrediction.foreach { case (x1, x2) =>
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

    // Sample training dataset: (feature1, feature2, label)
    val trainingData = Seq(
      (1.0, 2.0, 5.0),
      (2.0, 3.0, 7.0),
      (3.0, 4.0, 9.0),
      (4.0, 5.0, 11.0),
      (5.0, 6.0, 13.0)
    )

    // Sample test dataset: (feature1, feature2, label)
    val testData = Seq(
      (1.5, 2.5, 6.0),
      (2.5, 3.5, 8.0),
      (3.5, 4.5, 10.0),
      (4.5, 5.5, 12.0)
    )

    // Initialize weights and bias with small random values
    var weights = Array.fill(2)(Random.nextDouble() * 0.01)
    var bias = Random.nextDouble() * 0.01

    // Prediction function: returns continuous output
    def predict(features: (Double, Double)): Double = {
      weights(0) * features._1 + weights(1) * features._2 + bias
    }

    // Training loop using Gradient Descent
    for (_ <- 1 to numIterations) {
      var dw = Array(0.0, 0.0)
      var db = 0.0

      // Compute gradients
      for ((x1, x2, y) <- trainingData) {
        val prediction = predict((x1, x2))
        val error = prediction - y

        // Gradient of MSE with respect to weights and bias
        dw(0) += 2 * error * x1
        dw(1) += 2 * error * x2
        db += 2 * error
      }

      // Update weights and bias
      for (i <- weights.indices) {
        weights(i) -= learningRate * dw(i) / trainingData.length
      }
      bias -= learningRate * db / trainingData.length
    }

    // Final weights and bias
    println(s"Final weights: ${weights.mkString(", ")}")
    println(f"Final bias: $bias%.4f")

    // Function to calculate Mean Squared Error (MSE)
    def calculateMSE(data: Seq[(Double, Double, Double)]): Double = {
      val errors = data.map { case (x1, x2, label) =>
        val prediction = predict((x1, x2))
        Math.pow(prediction - label, 2)
      }
      errors.sum / data.length
    }

    // Calculate Mean Squared Error on the training data
    val trainingMSE = calculateMSE(trainingData)
    println(f"Mean Squared Error on training data: $trainingMSE%.4f")

    // Calculate Mean Squared Error on the test data
    val testMSE = calculateMSE(testData)
    println(f"Mean Squared Error on test data: $testMSE%.4f")

    // Predict on new data
    val testDataForPrediction = Seq((1.0, 1.5), (3.0, 2.5))
    testDataForPrediction.foreach { case (x1, x2) =>
      val prediction = predict((x1, x2))
      println(f"Prediction for ($x1, $x2): $prediction%.4f")
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

