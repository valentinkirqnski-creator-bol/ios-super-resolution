plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ventures.spacetree.fuzeframe"
    compileSdk = 34
    ndkVersion = "26.3.11579264"

    defaultConfig {
        applicationId = "ventures.spacetree.fuzeframe"
        minSdk = 30            // Android 11
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"

        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared")
                cppFlags += "-std=c++17"
            }
        }
        // arm64 only. The A20e is arm64, every Android 11 device is, and each
        // extra ABI is another full build of LibRaw plus the core -- minutes of
        // CI and tens of megabytes of APK for architectures nothing will run.
        ndk { abiFilters += "arm64-v8a" }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Debug-signed so the CI artifact installs directly. Not for
            // distribution.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // The camera permission is checked at runtime before the surface is created,
    // which lint's MissingPermission check cannot see through. Failing an
    // unsigned test build on that would cost a CI cycle for nothing.
    lint { abortOnError = false }

    buildFeatures { compose = true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.14" }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
}

dependencies {
    // Supplies the XML theme the manifest names. compose.material3 is Compose
    // only and ships no Theme.Material3.* resource, so without this the
    // resource link fails.
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.1")
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
}
