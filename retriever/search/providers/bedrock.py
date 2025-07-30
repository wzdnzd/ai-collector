#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AWS Bedrock provider implementation with manual SigV4 signing.
"""

import hashlib
import hmac
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from constants import DEFAULT_QUESTION
from models import CheckResult, Condition, ErrorReason

from logger import get_provider_logger
from utils import trim

from ..client import http_get
from .base import Provider

# Get Bedrock logger
logger = get_provider_logger()


class BedrockProvider(Provider):
    """AWS Bedrock provider with manual SigV4 authentication."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "anthropic.claude-3-7-sonnet-20250219-v1:0"
        base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"

        super().__init__(
            "bedrock",
            base_url,
            "/model/{model-id}/invoke",
            "/foundation-models",
            default_model,
            conditions,
            skip_search=skip_search,
        )

    def _parse_credentials(self, region: str, access_key: str, secret_key: str) -> tuple:
        """Parse and validate AWS credentials."""
        # Use default region if not provided
        region = trim(region) or "us-east-1"
        access_key = trim(access_key)
        secret_key = trim(secret_key)

        if not access_key or not secret_key:
            logger.error("AWS credentials are required: access_key and secret_key")
            return None, None, None

        # Validate access key format (AKIA...)
        if not re.match(r"^AKIA[0-9A-Z]{16}$", access_key):
            logger.warning(f"Access key format may be invalid: {access_key[:8]}...")

        return region, access_key, secret_key

    def _validate_region(self, region: str) -> str:
        """Validate and normalize AWS region."""
        region = trim(region) or "us-east-1"

        # Basic region format validation (supports standard and gov cloud regions)
        if not re.match(r"^[a-z]{2}(-[a-z]+)*-[a-z]+-[0-9]+$", region):
            logger.warning(f"Region format may be invalid: {region}, using us-east-1")
            return "us-east-1"

        return region

    def _build_invoke_url(self, region: str, model_id: str) -> str:
        """Build Bedrock InvokeModel API URL."""
        base_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        return f"{base_url}/model/{model_id}/invoke"

    def _build_models_url(self, region: str) -> str:
        """Build Bedrock ListFoundationModels API URL."""
        base_url = f"https://bedrock.{region}.amazonaws.com"
        return f"{base_url}/foundation-models"

    def _build_canonical_request(self, method: str, url: str, headers: Dict[str, str], payload: str) -> tuple[str, str]:
        """Build AWS SigV4 canonical request."""
        # Parse URL components
        parsed = urllib.parse.urlparse(url)
        # AWS requires URI to be percent-encoded (except for unreserved characters)
        uri = urllib.parse.quote(parsed.path or "/", safe="/")
        query = parsed.query or ""

        # Sort query parameters
        if query:
            params = urllib.parse.parse_qsl(query, keep_blank_values=True)
            params.sort()
            query = urllib.parse.urlencode(params)

        # Build canonical headers
        canonical_headers = ""
        signed_headers = ""

        # Sort headers by lowercase key
        sorted_headers = sorted(headers.items(), key=lambda x: x[0].lower())
        header_names = []

        for key, value in sorted_headers:
            key_lower = key.lower()
            header_names.append(key_lower)
            canonical_headers += f"{key_lower}:{value.strip()}\n"

        signed_headers = ";".join(header_names)

        # Hash payload
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # Build canonical request
        content = f"{method}\n{uri}\n{query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

        logger.debug(f"Canonical request built for {method} {url}")
        return content, signed_headers

    def _create_string_to_sign(self, timestamp: str, region: str, service: str, canonical_request: str) -> str:
        """Create AWS SigV4 string to sign."""
        algorithm = "AWS4-HMAC-SHA256"
        date = timestamp[:8]  # YYYYMMDD
        scope = f"{date}/{region}/{service}/aws4_request"

        # Hash canonical request
        request_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()

        # Build string to sign
        content = f"{algorithm}\n{timestamp}\n{scope}\n{request_hash}"

        logger.debug(f"String to sign created for {service} in {region}")
        return content, scope

    def _calculate_signature(
        self, secret_key: str, timestamp: str, region: str, service: str, string_to_sign: str
    ) -> str:
        """Calculate AWS SigV4 signature."""
        date = timestamp[:8]  # YYYYMMDD

        # Create signing key
        k_date = hmac.new(f"AWS4{secret_key}".encode("utf-8"), date.encode("utf-8"), hashlib.sha256).digest()
        k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
        k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
        k_signing = hmac.new(k_service, "aws4_request".encode("utf-8"), hashlib.sha256).digest()

        # Calculate signature
        signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        logger.debug("AWS SigV4 signature calculated")
        return signature

    def _generate_auth_header(self, access_key: str, credential_scope: str, signed_headers: str, signature: str) -> str:
        """Generate AWS SigV4 authorization header."""
        auth_header = (
            f"AWS4-HMAC-SHA256 "
            f"Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        )

        logger.debug("Authorization header generated")
        return auth_header

    def _sign_request(
        self, method: str, url: str, region: str, service: str, access_key: str, secret_key: str, payload: str = ""
    ) -> Dict[str, str]:
        """Sign AWS request with SigV4."""
        # Generate timestamp
        now = datetime.now(datetime.timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")

        # Parse host from URL
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc

        # Build headers
        headers = {"Host": host, "X-Amz-Date": timestamp, "Content-Type": "application/json"}

        # Note: Content-Length is typically added by the HTTP client automatically
        # and is not included in AWS SigV4 signing by boto3, so we don't include it here

        # Build canonical request
        canonical_request, signed_headers = self._build_canonical_request(method, url, headers, payload)

        # Create string to sign
        text, scope = self._create_string_to_sign(timestamp, region, service, canonical_request)

        # Calculate signature
        signature = self._calculate_signature(secret_key, timestamp, region, service, text)

        # Generate authorization header
        auth_header = self._generate_auth_header(access_key, scope, signed_headers, signature)
        headers["Authorization"] = auth_header

        logger.debug(f"Request signed for {service} API")
        return headers

    def _send_request(self, method: str, url: str, headers: Dict[str, str], payload: str = "") -> tuple:
        """Send HTTP request and return status code and response."""
        try:
            if method == "GET":
                response = http_get(url=url, headers=headers, retries=2, timeout=30)
                if response:
                    return 200, response
                else:
                    return 500, "Request failed"
            elif method == "POST":
                # For POST requests, we need to implement our own request
                req = urllib.request.Request(url, data=payload.encode("utf-8"), headers=headers, method="POST")

                try:
                    with urllib.request.urlopen(req, timeout=30) as response:
                        return response.getcode(), response.read().decode("utf-8")
                except urllib.error.HTTPError as e:
                    error_body = e.read().decode("utf-8") if e.fp else str(e.reason)
                    return e.code, error_body
                except Exception as e:
                    return 500, str(e)
            else:
                return 400, f"Unsupported method: {method}"

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return 500, str(e)

    def _parse_response(self, code: int, response: str) -> CheckResult:
        """Parse AWS Bedrock API response."""
        if code == 200:
            try:
                # Try to parse JSON response
                data = json.loads(response)
                if "body" in data or "completion" in data or "content" in data:
                    logger.debug("Bedrock API call successful")
                    return CheckResult.ok()
                else:
                    logger.warning(f"Unexpected response format: {response[:100]}...")
                    return CheckResult.ok()  # Assume success if we got 200
            except json.JSONDecodeError:
                # Non-JSON response but 200 status - assume success
                logger.debug("Non-JSON response with 200 status, assuming success")
                return CheckResult.ok()
        else:
            return self._handle_error(code, response)

    def _handle_error(self, status_code: int, response_body: str) -> CheckResult:
        """Handle AWS Bedrock API errors."""
        response_body = trim(response_body)

        logger.debug(f"Handling error: {status_code} - {response_body[:200]}...")

        if status_code == 400:
            if "ValidationException" in response_body:
                if any(
                    keyword in response_body.lower()
                    for keyword in ["model identifier", "invalid model", "model not found"]
                ):
                    return CheckResult.fail(ErrorReason.NO_MODEL)
                else:
                    return CheckResult.fail(ErrorReason.BAD_REQUEST)
            elif "ModelNotReadyException" in response_body:
                return CheckResult.fail(ErrorReason.NO_MODEL)
            elif "ModelTimeoutException" in response_body:
                return CheckResult.fail(ErrorReason.SERVER_ERROR)
            elif "ModelErrorException" in response_body:
                return CheckResult.fail(ErrorReason.NO_MODEL)
            elif "ServiceQuotaExceededException" in response_body:
                return CheckResult.fail(ErrorReason.NO_QUOTA)
            else:
                return CheckResult.fail(ErrorReason.BAD_REQUEST)

        elif status_code == 401:
            if "UnrecognizedClientException" in response_body:
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif "InvalidSignatureException" in response_body:
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif "TokenRefreshRequired" in response_body:
                return CheckResult.fail(ErrorReason.EXPIRED_KEY)
            else:
                return CheckResult.fail(ErrorReason.INVALID_KEY)

        elif status_code == 403:
            if "AccessDeniedException" in response_body:
                if "not authorized to perform" in response_body:
                    return CheckResult.fail(ErrorReason.NO_ACCESS)
                elif "does not have permission" in response_body:
                    return CheckResult.fail(ErrorReason.NO_ACCESS)
                else:
                    return CheckResult.fail(ErrorReason.NO_ACCESS)
            elif "UnauthorizedOperation" in response_body:
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif "RequestExpired" in response_body:
                return CheckResult.fail(ErrorReason.EXPIRED_KEY)
            else:
                return CheckResult.fail(ErrorReason.NO_ACCESS)

        elif status_code == 404:
            if "ResourceNotFoundException" in response_body:
                return CheckResult.fail(ErrorReason.NO_MODEL)
            else:
                return CheckResult.fail(ErrorReason.NO_MODEL)

        elif status_code == 429:
            if "ThrottlingException" in response_body:
                return CheckResult.fail(ErrorReason.RATE_LIMITED)
            elif "TooManyRequestsException" in response_body:
                return CheckResult.fail(ErrorReason.RATE_LIMITED)
            else:
                return CheckResult.fail(ErrorReason.RATE_LIMITED)

        elif status_code >= 500:
            if "InternalServerException" in response_body:
                return CheckResult.fail(ErrorReason.SERVER_ERROR)
            elif "ServiceUnavailableException" in response_body:
                return CheckResult.fail(ErrorReason.SERVER_ERROR)
            else:
                return CheckResult.fail(ErrorReason.SERVER_ERROR)
        else:
            logger.warning(f"Unknown error status: {status_code}")
            return CheckResult.fail(ErrorReason.UNKNOWN)

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for AWS Bedrock API requests. Not used directly."""
        # This method is required by base class but not used in Bedrock
        # since we use SigV4 signing instead of simple token auth
        return {"Content-Type": "application/json"}

    def _build_test_payload(self, model_id: str) -> str:
        """Build appropriate test payload based on model type."""
        model = model_id.lower()

        if "anthropic" in model or "claude" in model:
            # Anthropic Claude models
            return json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": DEFAULT_QUESTION}],
                }
            )
        elif "amazon" in model or "titan" in model:
            # Amazon Titan models
            return json.dumps(
                {
                    "inputText": DEFAULT_QUESTION,
                    "textGenerationConfig": {"maxTokenCount": 10, "temperature": 0.1},
                }
            )
        elif "ai21" in model or "jurassic" in model:
            # AI21 Jurassic models
            return json.dumps({"prompt": DEFAULT_QUESTION, "maxTokens": 10, "temperature": 0.1})
        elif "cohere" in model or "command" in model:
            # Cohere Command models
            return json.dumps({"prompt": DEFAULT_QUESTION, "max_tokens": 10, "temperature": 0.1})
        elif "meta" in model or "llama" in model:
            # Meta Llama models
            return json.dumps({"prompt": DEFAULT_QUESTION, "max_gen_len": 10, "temperature": 0.1})
        else:
            # Default to Anthropic format for unknown models
            logger.debug(f"Unknown model type {model_id}, using Anthropic format")
            return json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": DEFAULT_QUESTION}],
                }
            )

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check if AWS credentials are valid by calling Bedrock API."""
        # Parse parameters: region=address, access_key=endpoint, secret_key=token
        region, access_key, secret_key = self._parse_credentials(address, endpoint, token)

        if not all([region, access_key, secret_key]):
            logger.error("Invalid AWS credentials provided")
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        # Validate region
        region = self._validate_region(region)

        # Use provided model or default
        model_id = trim(model) or self.default_model

        # Build request URL
        url = self._build_invoke_url(region, model_id)

        # Prepare request payload based on model type
        payload = self._build_test_payload(model_id)

        try:
            # Sign request
            headers = self._sign_request(
                method="POST",
                url=url,
                region=region,
                service="bedrock-runtime",
                access_key=access_key,
                secret_key=secret_key,
                payload=payload,
            )

            # Send request
            status_code, response_body = self._send_request("POST", url, headers, payload)

            # Parse response
            result = self._parse_response(status_code, response_body)

            if result.available:
                logger.info(f"Bedrock credentials validated successfully for region {region} with model {model_id}")
            else:
                logger.warning(
                    f"Bedrock credentials validation failed: {result.reason} (region: {region}, model: {model_id})"
                )

            return result

        except Exception as e:
            logger.error(f"Bedrock check failed: {e}")
            return CheckResult.fail(ErrorReason.UNKNOWN)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available models from AWS Bedrock."""
        # Parse parameters: region=address, access_key=endpoint, secret_key=token
        region, access_key, secret_key = self._parse_credentials(address, endpoint, token)

        if not all([region, access_key, secret_key]):
            logger.error("Invalid AWS credentials provided for model listing")
            return []

        # Validate region
        region = self._validate_region(region)

        # Build request URL
        url = self._build_models_url(region)

        try:
            # Sign request (GET request, no payload)
            headers = self._sign_request(
                method="GET",
                url=url,
                region=region,
                service="bedrock",
                access_key=access_key,
                secret_key=secret_key,
                payload="",
            )

            # Send request
            code, response = self._send_request("GET", url, headers)

            if code == 200:
                try:
                    data = json.loads(response)
                    models = []

                    # Extract model IDs from response
                    if "modelSummaries" in data:
                        for model_info in data["modelSummaries"]:
                            if "modelId" in model_info:
                                models.append(model_info["modelId"])

                    logger.info(f"Retrieved {len(models)} models from Bedrock in region {region}")
                    return models

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse models response: {e}")
                    return []
            else:
                logger.error(f"Failed to list models: {code} - {response[:200]}...")
                return []

        except Exception as e:
            logger.error(f"Bedrock list_models failed: {e}")
            return []
            return []
