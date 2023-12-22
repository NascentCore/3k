FROM openjdk:8
ARG JAR_FILE=aiadmin/aiadmin-system/target/*.jar
COPY ${JAR_FILE} app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
