// import { Component } from '@angular/core';

// @Component({
//   selector: 'app-feedback',
//   templateUrl: './feedback.component.html',
//   styleUrls: ['./feedback.component.css']
// })
// export class FeedbackComponent {

// }
import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
@Component({
  selector: 'app-feedback',
  templateUrl: './feedback.component.html',
  styleUrls: ['./feedback.component.css']
})
export class FeedbackComponent implements OnInit {
  storedFeedbacks: any[] = [];
  constructor(private http: HttpClient, private router: Router) { }
 

  ngOnInit(): void {
    this.loadStoredFeedbacks();
  }
 
  submitFeedback() {
    const nameInput = document.getElementById('name') as HTMLInputElement;
    const customeridInput = document.getElementById('customerid') as HTMLInputElement;
    const feedbackInput = document.getElementById('feedback') as HTMLTextAreaElement;

    const feedbackData = {
      name: nameInput.value,
      customerid: customeridInput.value,
      feedback: feedbackInput.value
    };

    // Save feedbackData to localStorage
    let feedbacksString = localStorage.getItem('feedbacks');
    let feedbacks: any[] = [];
    if (feedbacksString) {
      feedbacks = JSON.parse(feedbacksString);
    }
    feedbacks.push(feedbackData);
    localStorage.setItem('feedbacks', JSON.stringify(feedbacks));

    alert('Feedback submitted successfully!');
    nameInput.value = '';
    customeridInput.value = '';
    feedbackInput.value = '';
    this.loadStoredFeedbacks();
  }

  loadStoredFeedbacks() {
    const feedbacksString = localStorage.getItem('feedbacks');
    this.storedFeedbacks = feedbacksString ? JSON.parse(feedbacksString) : [];
  }

  onFinish(): void{
    this.router.navigate(['/thankyou']); 
  }
}
