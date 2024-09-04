import scala.util.Random

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    val learningRate = 0.01
    val numIterations = 1000

    // Sample dataset: (feature1, feature2, label)
    val data = Seq(
      (0.5, 0.7, 0),
      (1.5, 2.0, 0),
      (3.0, 3.5, 1),
      (2.5, 2.7, 1),
      (3.5, 4.0, 1)
    )

    // Initialize weights and bias
    var weights = Array.fill(2)(Random.nextDouble())
    var bias = Random.nextDouble()

    // Sigmoid function
    def sigmoid(z: Double): Double = 1.0 / (1.0 + Math.exp(-z))

    // Prediction function
    def predict(features: (Double, Double)): Double = {
      val linearModel = weights(0) * features._1 + weights(1) * features._2 + bias
      sigmoid(linearModel)
    }

    // Training loop
    for (_ <- 0 until numIterations) {
      var dw = Array(0.0, 0.0)
      var db = 0.0

      // Gradient descent
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
    }

    // Final weights and bias
    println(s"Final weights: ${weights.mkString(", ")}")
    println(s"Final bias: $bias")

    // Calculate accuracy on the training data
    val predictions = data.map { case (x1, x2, label) =>
      val prob = predict((x1, x2))
      val predictedLabel = if (prob >= 0.5) 1 else 0
      (predictedLabel, label)
    }

    val accuracy = predictions.count { case (predicted, actual) => predicted == actual }.toDouble / data.length
    println(f"Overall accuracy on training data: $accuracy%.2f")

    // Predict on new data
    val testData = Seq((1.0, 1.5), (3.0, 2.5))
    testData.foreach { case (x1, x2) =>
      val prob = predict((x1, x2))
      println(s"Prediction for ($x1, $x2): ${if (prob >= 0.5) 1 else 0} (probability: $prob)")
    }
  }
}

