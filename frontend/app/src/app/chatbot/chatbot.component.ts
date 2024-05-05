
import { HttpClient } from '@angular/common/http';
import { Component, AfterViewInit } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent implements AfterViewInit {
  message: string | undefined;
  constructor(private http: HttpClient, private router: Router) { }
  
  ngOnInit(): void {}

  userMessage = [
    ["can i buy"]
  ];

  // Function to handle sending messages
  sendMessage() {
    const inputField = document.getElementById("input") as HTMLInputElement;
    let input = inputField.value.trim();
    
    // Check if the input field is empty
    if (input === '') {
      this.addChat("Input field should not be empty.", "Please enter a valid query.");
    } else {
      // If input is not empty, send the message
      this.output(input);
    }
    
    inputField.value = "";
  }
  
  // Function to handle sending placeholder message on click
  ngAfterViewInit() {
    const inputField = document.getElementById("input") as HTMLInputElement;
    inputField.addEventListener("click", () => {
      let input = inputField.placeholder.trim();
      if (input !== '') {
        this.output(input);
      }
    });
  }
  
  // Function to send request to backend and handle response
  output(input: string) {
    this.addChat(input, "Fetching overall trend...");

    // Send GET request to backend for overall trend only if input is not empty
    this.http.get<any>('http://127.0.0.1:5000/trend').subscribe(
      (response) => {
        const overallTrend = response.message;

        // Process the overall trend to generate appropriate response
        const trendResponse = this.processOverallTrend(overallTrend);
        this.addChat(overallTrend, trendResponse); // Add the response to the UI
      },
      (error) => {
        console.error("Error occurred while fetching overall trend:", error);
      }
    );
  }
  
  // Function to process overall trend and generate appropriate response
  processOverallTrend(overallTrend: string): string {
    // Check the overall trend and generate the appropriate response
    if (overallTrend.includes("increasing")) {
      return "You can buy later.";
    } else if (overallTrend.includes("decreasing")) {
      return "You can buy now.";
    } else {
      return "Trend information not available.";
    }
  }
  
  // Function to add chat messages to the UI
  addChat(input: any, product: any) {
    const mainDiv = document.getElementById("message-section");
    let userDiv = document.createElement("div");
    userDiv.id = "user";
    userDiv.classList.add("message");
    userDiv.innerHTML = `<span id="user-response">${input}</span>`;
    mainDiv?.appendChild(userDiv);
  
    let botDiv = document.createElement("div");
    botDiv.id = "bot";
    botDiv.classList.add("message");
    botDiv.innerHTML = `<span id="bot-response">${product}</span>`;
    mainDiv?.appendChild(botDiv);
    var scroll = document.getElementById("message-section");
    scroll?.scrollTo(0, scroll.scrollHeight);
  }

  // Function to navigate to feedback page
  onFeedback() {
    this.router.navigate(['/feedback']);
  }
}
