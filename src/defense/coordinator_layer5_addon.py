        
    def run_galaxy_defense(
        self,
        galaxy_updates: Dict[int, List[torch.Tensor]],
        client_assignments: Dict[int, int],
        round_number: int
    ) -> Dict:
        """Run Layer 5: Galaxy-level defense
        
        Args:
            galaxy_updates: Dict mapping galaxy_id -> aggregated gradients
            client_assignments: Client-to-galaxy mapping
            round_number: Current FL round
            
        Returns:
            Layer 5 defense results
        """
        client_reputations = self.layer4.get_all_reputations()
        
        layer5_results = self.layer5.run_defense(
            galaxy_updates=galaxy_updates,
            client_assignments=client_assignments,
            client_reputations=client_reputations
        )
        
        # Log quarantine/ban decisions for flagged galaxies
        for galaxy_id in layer5_results['flagged_galaxies']:
            anomaly_report = layer5_results['anomaly_reports'][galaxy_id]
            
            # Get clients in this galaxy
            galaxy_clients = [
                cid for cid, gid in client_assignments.items()
                if gid == galaxy_id
            ]
            
            # Log evidence for each low-reputation client
            for client_id in galaxy_clients:
                rep = client_reputations.get(client_id, 1.0)
                
                if self.layer4.is_quarantined(client_id):
                    # Log quarantine with galaxy-level evidence
                    self.forensic_logger.log_quarantine(
                        client_id=client_id,
                        round_number=round_number,
                        commitment_hash="",  # To be filled by pipeline
                        merkle_proof=[],  # To be filled by pipeline
                        merkle_root="",  # To be filled by pipeline
                        layer_results={
                            'layer1_failed': False,
                            'layer2_flags': [],
                            'layer3_rejected': False,
                            'layer4_reputation': rep,
                            'layer5_galaxy_flagged': True,
                            'layer5_failed_checks': anomaly_report.failed_checks
                        },
                        metadata={
                            'galaxy_id': galaxy_id,
                            'galaxy_norm_deviation': anomaly_report.norm_deviation_score,
                            'galaxy_direction_similarity': anomaly_report.direction_similarity,
                            'galaxy_consistency': anomaly_report.cross_galaxy_consistency
                        }
                    )
                elif self.layer4.is_banned(client_id):
                    # Log permanent ban
                    self.forensic_logger.log_ban(
                        client_id=client_id,
                        round_number=round_number,
                        commitment_hash="",
                        merkle_proof=[],
                        merkle_root="",
                        layer_results={
                            'layer1_failed': False,
                            'layer2_flags': [],
                            'layer3_rejected': False,
                            'layer4_reputation': rep,
                            'layer5_galaxy_flagged': True,
                            'layer5_failed_checks': anomaly_report.failed_checks
                        },
                        metadata={'galaxy_id': galaxy_id}
                    )
        
        return layer5_results
