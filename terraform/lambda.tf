# =========================================================================
# Alternative Serverless Deployment: AWS Lambda Container
# =========================================================================

# CloudWatch Log Group for Lambda
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.app_name}-lambda"
  retention_in_days = 7
}

# IAM Role for Lambda execution
resource "aws_iam_role" "lambda_exec" {
  name = "${var.app_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Attach basic execution policy (cloudwatch logging)
resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# AWS Lambda Function configured from ECR Docker image
resource "aws_lambda_function" "api" {
  function_name = "${var.app_name}-lambda"
  role          = aws_iam_role.lambda_exec.arn
  image_uri     = "${aws_ecr_repository.api.repository_url}:latest"
  package_type  = "Image"
  
  # Configure CPU allocation and compute performance via Memory size
  memory_size   = var.lambda_memory_size
  timeout       = 30 # 30-second timeout to handle potential LLM/RAG API call duration

  # Deploy Lambda on Graviton2 (ARM64) for ~20% lower cost and higher performance
  architectures = ["arm64"]

  environment {
    variables = {
      MODEL_BACKEND  = "onnx" # Production ONNX backend
      MODEL_NAME     = "BAAI/bge-m3"
      INDEX_PATH     = "embeddings.joblib"
      SLA_LATENCY_MS = "300.0"
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy_attachment.lambda_policy
  ]
}

# =========================================================================
# API Gateway HTTP API (V2) - Serverless Router
# =========================================================================

resource "aws_apigatewayv2_api" "http_api" {
  name          = "${var.app_name}-http-api"
  protocol_type = "HTTP"
  description   = "Serverless Gateway for Sigma RAG Model Serving API"
}

# API Gateway integration with Lambda function
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.http_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.arn
  payload_format_version = "2.0"
}

# Route everything ($default) to Lambda
resource "aws_apigatewayv2_route" "default_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# Default stage with automatic deployment
resource "aws_apigatewayv2_stage" "default_stage" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true
  
  # Enable request logging for API Gateway
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.apigateway.arn
    format          = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
      latency        = "$context.responseLatency"
    })
  }
}

# CloudWatch log group for API Gateway
resource "aws_cloudwatch_log_group" "apigateway" {
  name              = "/aws/apigateway/${var.app_name}-api"
  retention_in_days = 7
}

# IAM permission allowing API Gateway to invoke the Lambda function
resource "aws_lambda_permission" "apigw_lambda" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}

# Outputs
output "lambda_api_endpoint" {
  value       = aws_apigatewayv2_stage.default_stage.invoke_url
  description = "The HTTP API Gateway URL to invoke the serverless Lambda function"
}
